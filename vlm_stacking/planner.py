"""VLM-backed symbolic planning and validation for stacking."""

import base64
import json
import os
import urllib.error
import urllib.request
import tempfile

import ollama


class StackPlanError(ValueError):
  """Raised when a VLM plan cannot be safely mapped to the scene."""


STACK_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "placements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_id": {"type": "string"},
                    "slot_id": {"type": "string"},
                },
                "required": ["object_id", "slot_id"],
                "additionalProperties": False,
            },
            "description": "Objects assigned to target slots, in safe build order.",
        },
        "summary": {"type": "string"},
    },
    "required": ["placements", "summary"],
    "additionalProperties": False,
}


class OpenAIResponsesStackPlanner:
  """Ask a vision-capable Responses API model for an object ordering.

  The API is called with the Python standard library so this module does not
  add a dependency that conflicts with Ravens' older TensorFlow environment.
  Set OPENAI_API_KEY and optionally OPENAI_VLM_MODEL before use.
  """

  def __init__(self, model=None, timeout=60):
    self.model = model or os.environ.get("OPENAI_VLM_MODEL", "gpt-4o-mini")
    self.timeout = timeout

  def plan(self, instruction, image_bytes, objects, target_slots=None):
    """Return a validated plan for the current image and object catalog.

    Args:
      instruction: User's stacking instruction.
      image_bytes: JPEG or PNG bytes showing the current scene.
      objects: List of dictionaries containing stable `object_id` values and
        measured properties such as dimensions, color, and shape.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
      raise RuntimeError("Set OPENAI_API_KEY to call the VLM planner.")
    if not image_bytes:
      raise ValueError("A non-empty scene image is required.")
    if not objects:
      raise StackPlanError("No pickable objects were found in the scene.")
    if target_slots is None:
      raise StackPlanError("Target slot descriptions are required for planning.")

    content_type = "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
      content_type = "image/jpeg"
    image_url = "data:%s;base64,%s" % (
        content_type, base64.b64encode(image_bytes).decode("ascii"))

    prompt = (
        self._build_prompt(instruction, objects, target_slots))
    payload = {
        "model": self.model,
        "store": False,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_url,
                 "detail": "high"},
            ],
        }],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "stack_plan",
                "strict": True,
                "schema": STACK_PLAN_SCHEMA,
            }
        },
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST")
    try:
      with urllib.request.urlopen(request, timeout=self.timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
      detail = error.read().decode("utf-8", errors="replace")
      raise RuntimeError("VLM API returned HTTP %s: %s" % (
          error.code, detail))
    except urllib.error.URLError as error:
      raise RuntimeError("Could not reach the VLM API: %s" % error)

    text = _response_output_text(body)
    try:
      plan = json.loads(text)
    except (TypeError, ValueError) as error:
      raise StackPlanError("VLM response was not valid plan JSON: %s" % error)
    validate_plan(plan, objects, target_slots)
    return plan

  @staticmethod
  def _build_prompt(instruction, objects, target_slots, previous_output=None,
                    repair_error=None, repair_kind=None):
    if target_slots is None:
      target_slots = []
    return (
        "Plan a stable pyramid matching the user's instruction. Assign each "
        "catalog object to exactly one available target slot. Use the readable "
        "color field as the authoritative object color. Respect the slot's "
        "layer and support requirements. Build in dependency order: base slots, "
        "then middle slots, then apex. Target slot IDs are labeled in the image; "
        "use those labels to interpret spatial instructions. Use only supplied object and slot IDs; "
        "do not return coordinates or invent geometry. The robot will move each "
        "assigned object to the simulator pose for its slot.\n\n"
        "User instruction: %s\n\n"
        "Measured object catalog (dimensions are meters):\n%s" % (
            instruction, json.dumps(objects, sort_keys=True)) +
        "\n\nAvailable target slots (slot order is base to apex):\n%s\n\n"
        "Use each slot's allowed_colors as a hard task constraint.\n"
        "Return only JSON with this shape: {\"placements\":["
        "{\"object_id\":\"...\",\"slot_id\":\"...\"}],"
        "\"summary\":\"...\"}. Include every object and every slot once."
        % json.dumps(target_slots, sort_keys=True))

class OllamaStackPlanner:
    """Ask a local vision-capable Ollama model for an object ordering.

    Requires a running Ollama server with a vision-language model pulled
    (e.g., `ollama pull qwen2.5vl`). No API key needed.
    """

    def __init__(self, model=None, max_retries=3):
        self.model = model or os.environ.get("OLLAMA_VLM_MODEL", "qwen2.5vl:7b")
        self.max_retries = max(0, int(max_retries))

    def _call_llm(self, prompt, tmp_path):
      try:
          response = ollama.chat(
              model=self.model,
              messages=[
                  {
                      "role": "user",
                      "content": prompt,
                      "images": [tmp_path],
                  }
              ],
          )
      except Exception as error:
          raise RuntimeError(
              "Ollama chat failed (is the server running and '%s' pulled?): "
              "%s" % (self.model, error)
          )
      finally:
          # Clean up the temporary image file regardless of outcome.
          try:
              os.unlink(tmp_path)
          except OSError:
              pass
      return response

    def plan(self, instruction, image_bytes, objects, target_slots=None):
        """Return a validated plan for the current image and object catalog.

        Args:
            instruction: User's stacking instruction.
            image_bytes: JPEG or PNG bytes showing the current scene.
            objects: List of dictionaries containing stable `object_id` values
                and measured properties such as dimensions, color, and shape.

        Returns:
            A plan that passes `validate_plan`.

        Raises:
            ValueError: If image_bytes is empty.
            StackPlanError: If no objects are provided or the model response
                remains unparseable after the configured retries.
            StackPlanError: If the plan still fails validation after retries.
        """
        if not image_bytes:
            raise ValueError("A non-empty scene image is required.")
        if not objects:
            raise StackPlanError("No pickable objects were found in the scene.")
        if target_slots is None:
            raise StackPlanError("Target slot descriptions are required for planning.")

        previous_output = None
        repair_error = None
        repair_kind = None
        for attempt in range(self.max_retries + 1):
            prompt = self._build_prompt(
                instruction, objects, target_slots, previous_output, repair_error,
                repair_kind)
            try:
                plan = self._generate_plan(prompt, image_bytes)
            except StackPlanError as error:
                previous_output = str(error)
                repair_error = str(error)
                repair_kind = "JSON parsing"
                if attempt >= self.max_retries:
                    raise StackPlanError(
                        "Could not produce a valid plan after %d retries "
                        "(%d attempts). Last JSON parsing error: %s" % (
                            self.max_retries, self.max_retries + 1, error))
                continue

            try:
                validate_plan(plan, objects, target_slots)
            except StackPlanError as error:
                previous_output = json.dumps(plan, ensure_ascii=False)
                repair_error = str(error)
                repair_kind = "plan validation"
                if attempt >= self.max_retries:
                    repaired = repair_plan_with_eligibility(
                        plan, objects, target_slots)
                    if repaired is not None:
                        repaired["summary"] = (
                            "VLM assignment corrected deterministically to "
                            "satisfy the task's color and layer constraints. "
                            + str(plan.get("summary", "")))
                        return validate_plan(repaired, objects, target_slots)
                    raise StackPlanError(
                        "Could not produce a valid plan after %d retries "
                        "(%d attempts). Last validation error: %s" % (
                            self.max_retries, self.max_retries + 1, error))
                continue
            return plan

    @staticmethod
    def _build_prompt(instruction, objects, target_slots=None, previous_output=None,
                      repair_error=None, repair_kind=None):
        layer_names = {0: "base", 1: "middle", 2: "apex"}
        grouped_rules = {}
        for slot in target_slots or []:
            layer_index = int(slot.get("layer_index", 0))
            group = grouped_rules.setdefault(layer_index, {
                "name": layer_names.get(layer_index, "layer %d" % layer_index),
                "slots": [], "colors": set(), "object_ids": set()})
            group["slots"].append(slot["slot_id"])
            group["colors"].update(slot.get("allowed_colors", []))
            group["object_ids"].update(slot.get("allowed_object_ids", []))
        color_rules = "\n".join(
            "- %s slots (%s): objects %s, colors %s only." % (
                group["name"], ", ".join(group["slots"]),
                json.dumps(sorted(group["object_ids"])),
                ", ".join(sorted(group["colors"])))
            for _, group in sorted(grouped_rules.items()))
        prompt = (
              "Plan a stable pyramid matching the user's instruction. Assign "
              "each catalog object to exactly one target slot. Each catalog entry may include "
              "a human-readable color field; use it to identify colors and treat "
              "it as authoritative; numerical RGB is secondary. Follow each "
              "slot's layer and support constraints. The color-to-layer rules "
              "below are hard constraints. Never assign an object to a layer "
              "that does not list its color. Place base objects before middle "
              "objects and middle objects before the apex. "
              "Target slot IDs are labeled in the scene image; use those labels "
              "to interpret spatial instructions. Do not return coordinates; "
              "the robot maps slot IDs to measured "
              "task target poses. Treat each slot's allowed_colors as a hard "
              "constraint.\n\n"
              "Required color-to-layer rules:\n%s\n\n"
              "User instruction: %s\n\n"
              "Measured object catalog:\n%s\n\n"
              "Available target slots, listed in build order:\n%s\n\n"
              "Return ONLY JSON in this format: {\"placements\":["
              "{\"object_id\":\"...\",\"slot_id\":\"...\"}],"
              "\"summary\":\"...\"}. Assign every object and every slot once."
              % (color_rules, instruction, json.dumps(objects, sort_keys=True),
                 json.dumps(target_slots or [], sort_keys=True)))
        if previous_output is not None or repair_error is not None:
            required_ids = [str(obj["object_id"]) for obj in objects]
            prompt += (
                "\n\nYour previous response failed %s checks. Correct "
                "the response using the error below. The required object IDs "
                "are %s; include every one exactly once in placements, and "
                "do not include any other IDs. Required slot IDs are %s. "
                "Use every slot ID "
                "exactly once, obey allowed_colors and build layers in order. "
                "Return only the complete JSON "
                "object.\nPrevious response or plan:\n%s\nError:\n%s\n" % (
                    repair_kind or "plan",
                    json.dumps(required_ids),
                    json.dumps([slot["slot_id"] for slot in target_slots or []]),
                    previous_output or "(empty response)",
                    repair_error or "The plan did not pass validation."))
        return prompt

    def _generate_plan(self, prompt, image_bytes):
        # Ollama's Python client accepts either a file path or raw bytes for
        # the `images` field. Writing a temporary image is portable across
        # ollama-python versions.
        suffix = ".jpg" if image_bytes[:3] == b"\xff\xd8\xff" else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        response = self._call_llm(prompt, tmp_path)
        try:
            text = response["message"]["content"]
            text = _extract_json(text)
            return json.loads(text)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raw_output = (response.get("message", {}).get("content", "")
                          if isinstance(response, dict) else repr(response))
            raise StackPlanError(
                "VLM response was not a parseable plan JSON object: %s\n"
                "Raw output: %s" % (error, raw_output))


class OllamaGraspPlanner:
    """Ask a local vision-language model for a bounded top-down grasp command."""

    def __init__(self, model=None, max_retries=2):
        self.model = model or os.environ.get("OLLAMA_VLM_MODEL", "qwen2.5vl:7b")
        self.max_retries = max(0, int(max_retries))

    def plan(self, instruction, image_bytes, object_state, object_catalog):
        if not image_bytes:
            raise ValueError("A non-empty scene image is required for grasping.")
        if object_state is None:
            raise StackPlanError("The selected object has no measured state.")

        previous_error = None
        for attempt in range(self.max_retries + 1):
            prompt = self._build_prompt(
                instruction, object_state, object_catalog, previous_error)
            suffix = ".jpg" if image_bytes[:3] == b"\xff\xd8\xff" else ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt,
                               "images": [tmp_path]}])
            except Exception as error:
                raise RuntimeError(
                    "Ollama grasp request failed (is the server running and "
                    "'%s' pulled?): %s" % (self.model, error))
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            try:
                content = response["message"]["content"]
                command = json.loads(_extract_json(content))
                return self._validate(command)
            except (AttributeError, KeyError, TypeError, ValueError,
                    StackPlanError) as error:
                previous_error = str(error)
                if attempt == self.max_retries:
                    raise StackPlanError(
                        "Could not produce a valid grasp command after %d "
                        "attempts: %s" % (self.max_retries + 1, error))

    def plan_all(self, instruction, image_bytes, object_catalog):
        """Plan one grasp per visible object in a single VLM request."""
        if not image_bytes:
            raise ValueError("A non-empty scene image is required for grasping.")
        if not object_catalog:
            raise StackPlanError("No visible objects were provided for grasping.")

        previous_error = None
        required_ids = [str(item["object_id"]) for item in object_catalog]
        for attempt in range(self.max_retries + 1):
            prompt = self._build_batch_prompt(
                instruction, object_catalog, previous_error)
            try:
                response = self._generate_json(prompt, image_bytes)
                if not isinstance(response, dict) or not isinstance(
                        response.get("grasps"), list):
                    raise StackPlanError("Batch grasp response must contain a grasps list.")
                commands = {}
                for row in response["grasps"]:
                    if not isinstance(row, dict) or "object_id" not in row:
                        raise StackPlanError("Every grasp must include an object_id.")
                    object_id = str(row["object_id"])
                    if object_id not in required_ids or object_id in commands:
                        raise StackPlanError(
                            "Grasp response has an unknown or repeated object_id: %s" %
                            object_id)
                    commands[object_id] = {
                        "object_id": object_id,
                        **self._validate(row),
                    }
                if set(commands) != set(required_ids):
                    raise StackPlanError(
                        "Batch grasp response must include every visible object exactly once.")
                return commands
            except StackPlanError as error:
                previous_error = str(error)
                if attempt == self.max_retries:
                    raise StackPlanError(
                        "Could not produce valid grasps for all objects after "
                        "%d attempts: %s" % (self.max_retries + 1, error))

    def _generate_json(self, prompt, image_bytes):
        suffix = ".jpg" if image_bytes[:3] == b"\xff\xd8\xff" else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt,
                           "images": [tmp_path]}])
        except Exception as error:
            raise RuntimeError(
                "Ollama grasp request failed (is the server running and "
                "'%s' pulled?): %s" % (self.model, error))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        try:
            content = response["message"]["content"]
            return json.loads(_extract_json(content))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise StackPlanError("Grasp response was not valid JSON: %s" % error)

    @staticmethod
    def _build_batch_prompt(instruction, object_catalog, previous_error):
        prompt = (
            "Choose one stable top-down two-finger grasp for every listed "
            "object in this scene. The robot has a parallel-jaw gripper; the "
            "simulator handles jaw closure. Use the image and each object's "
            "measured dynamics (dimensions, pose, mass, and velocity). Return "
            "one command per object ID. For each command, use "
            "grasp_yaw_degrees in [0, 180), and "
            "local_offset_x_fraction and local_offset_y_fraction each in "
            "[-0.15, 0.15]. Offsets are fractions of that object's local XY "
            "dimensions. Prefer centered grasps. Return JSON only with this "
            "shape: {\"grasps\":[{\"object_id\":\"...\","
            "\"grasp_yaw_degrees\":0,\"local_offset_x_fraction\":0,"
            "\"local_offset_y_fraction\":0}]}\n\n"
            "User instruction: %s\n\nVisible object catalog:\n%s" % (
                instruction, json.dumps(object_catalog, sort_keys=True)))
        if previous_error:
            prompt += "\n\nCorrect the previous invalid response: %s" % (
                previous_error)
        return prompt

    @staticmethod
    def _build_prompt(instruction, object_state, object_catalog, previous_error):
        prompt = (
            "Choose a stable top-down two-finger grasp for the selected "
            "object in this scene. The robot has a parallel-jaw gripper and "
            "will use the returned yaw and small local offset to approach the "
            "object; the simulator handles jaw closure. Use the image and "
            "measured object dynamics (dimensions, pose, mass, and velocity). "
            "Keep the grasp on the selected object. Return JSON only with "
            "grasp_yaw_degrees in [0, 180), and local_offset_x_fraction and "
            "local_offset_y_fraction each in [-0.15, 0.15]. Offsets are "
            "fractions of the selected object's local XY dimensions. Use zero "
            "offset for a centered grasp. Do not return prose or motor commands.\n\n"
            "User instruction: %s\n\nSelected object state:\n%s\n\n"
            "Visible object catalog:\n%s\n\n"
            "Return exactly this JSON shape: {\"grasp_yaw_degrees\": 0, "
            "\"local_offset_x_fraction\": 0, "
            "\"local_offset_y_fraction\": 0}." % (
                instruction, json.dumps(object_state, sort_keys=True),
                json.dumps(object_catalog, sort_keys=True)))
        if previous_error:
            prompt += "\n\nThe previous response was invalid: %s. Correct it." % (
                previous_error)
        return prompt

    @staticmethod
    def _validate(command):
        if not isinstance(command, dict):
            raise StackPlanError("Grasp command must be a JSON object.")
        required = ("grasp_yaw_degrees", "local_offset_x_fraction",
                    "local_offset_y_fraction")
        if any(key not in command for key in required):
            raise StackPlanError("Grasp command is missing required fields.")
        try:
            yaw, offset_x, offset_y = (float(command[key]) for key in required)
        except (TypeError, ValueError) as error:
            raise StackPlanError("Grasp parameters must be numeric: %s" % error)
        if not 0.0 <= yaw < 180.0:
            raise StackPlanError("grasp_yaw_degrees must be in [0, 180).")
        if not -0.15 <= offset_x <= 0.15 or not -0.15 <= offset_y <= 0.15:
            raise StackPlanError("Local grasp offsets must be in [-0.15, 0.15].")
        return {"grasp_yaw_degrees": yaw,
                "local_offset_x_fraction": offset_x,
                "local_offset_y_fraction": offset_y}

def _extract_json(text):
  """Strip markdown code fences and leading prose to isolate JSON."""
  stripped = text.strip()
  if stripped.startswith("```"):
    first_newline = stripped.index("\n")
    stripped = stripped[first_newline + 1 :]
    if stripped.rstrip().endswith("```"):
      stripped = stripped.rstrip()[:-3]
    return stripped.strip()

  brace = stripped.find("{")
  if brace != -1:
    depth = 0
    for i in range(brace, len(stripped)):
      if stripped[i] == "{":
        depth += 1
      elif stripped[i] == "}":
        depth -= 1
        if depth == 0:
          return stripped[brace : i + 1]
  return stripped

def _response_output_text(response):
  """Extract output text from a Responses API response object."""
  chunks = []
  for item in response.get("output", []):
    if item.get("type") != "message":
      continue
    for part in item.get("content", []):
      if part.get("type") in ("output_text", "text"):
        chunks.append(part.get("text", ""))
  if not chunks:
    raise StackPlanError("VLM response did not contain a text plan.")
  return "\n".join(chunks)


def validate_plan(plan, objects, target_slots=None):
  """Validate one-to-one object and target-slot assignment.

  This checks symbolic coverage and task constraints. It is not a physical
  stability proof or a collision check.
  """
  if not isinstance(plan, dict) or not isinstance(plan.get("placements"), list):
    raise StackPlanError("Plan must contain a placements list.")
  ids = [str(obj["object_id"]) for obj in objects]
  placements = plan["placements"]
  order = [str(row.get("object_id")) for row in placements
           if isinstance(row, dict)]
  slot_order = [str(row.get("slot_id")) for row in placements
                if isinstance(row, dict)]
  if len(order) != len(placements) or len(slot_order) != len(placements):
    raise StackPlanError("Each placement must contain object_id and slot_id.")
  if len(order) != len(set(order)):
    raise StackPlanError("Plan repeats an object ID: %s" % order)
  if set(order) != set(ids):
    raise StackPlanError(
        "Plan object IDs must match the visible object catalog exactly. Output IDs %s, expected IDs %s" % (order, ids))
  if len(slot_order) != len(set(slot_order)):
    raise StackPlanError("Plan repeats a target slot: %s" % slot_order)
  expected_slots = [str(slot["slot_id"]) for slot in (target_slots or [])]
  if target_slots is not None and set(slot_order) != set(expected_slots):
    raise StackPlanError("Plan target slots must match available slots exactly. Output %s, expected %s" % (slot_order, expected_slots))
  slot_layers = {str(slot["slot_id"]): int(slot.get("layer_index", 0))
                 for slot in (target_slots or [])}
  if slot_layers and [slot_layers[s] for s in slot_order] != sorted(
      slot_layers[s] for s in slot_order):
    raise StackPlanError("Placements must be ordered base-to-apex by layer.")
  if not isinstance(plan.get("summary"), str):
    raise StackPlanError("Plan summary must be text.")
  if target_slots is not None:
    slots_by_id = {str(slot["slot_id"]): slot for slot in target_slots}
    colors_by_object = {str(obj["object_id"]): obj.get("color")
                        for obj in objects}
    for row in placements:
      slot = slots_by_id[str(row["slot_id"])]
      allowed_ids = slot.get("allowed_object_ids")
      if allowed_ids and str(row["object_id"]) not in {
          str(value) for value in allowed_ids}:
        raise StackPlanError(
            "Object %s is not eligible for slot %s; eligible object IDs are %s." % (
                row["object_id"], row["slot_id"], allowed_ids))
      allowed = slot.get("allowed_colors")
      color = colors_by_object.get(str(row["object_id"]))
      if allowed and color not in allowed:
        raise StackPlanError(
            "Object %s has color %s, which is not allowed in slot %s; "
            "allowed colors are %s." % (
                row["object_id"], color, row["slot_id"], allowed))
  return plan


def repair_plan_with_eligibility(plan, objects, target_slots):
  """Repair a complete assignment using hard task eligibility constraints."""
  if not isinstance(plan, dict) or not isinstance(plan.get("placements"), list):
    return None
  object_by_id = {str(obj["object_id"]): obj for obj in objects}
  slot_by_id = {str(slot["slot_id"]): slot for slot in target_slots or []}
  assignments = {}
  assigned_objects = set()
  for row in plan["placements"]:
    if not isinstance(row, dict):
      continue
    object_id, slot_id = str(row.get("object_id")), str(row.get("slot_id"))
    if object_id not in object_by_id or slot_id not in slot_by_id:
      continue
    slot = slot_by_id[slot_id]
    obj = object_by_id[object_id]
    allowed_ids = slot.get("allowed_object_ids")
    allowed_colors = slot.get("allowed_colors")
    if (object_id not in assigned_objects and
        (not allowed_ids or object_id in {str(value) for value in allowed_ids}) and
        (not allowed_colors or obj.get("color") in allowed_colors)):
      assignments[slot_id] = object_id
      assigned_objects.add(object_id)

  for slot_id, slot in slot_by_id.items():
    if slot_id in assignments:
      continue
    allowed_ids = ({str(value) for value in slot.get("allowed_object_ids", [])}
                   if slot.get("allowed_object_ids") else None)
    allowed_colors = slot.get("allowed_colors")
    object_id = next((str(obj["object_id"]) for obj in objects
                      if str(obj["object_id"]) not in assigned_objects and
                      (allowed_ids is None or
                       str(obj["object_id"]) in allowed_ids) and
                      (not allowed_colors or
                       obj.get("color") in allowed_colors)), None)
    if object_id is None:
      return None
    assignments[slot_id] = object_id
    assigned_objects.add(object_id)
  if assigned_objects != set(object_by_id) or len(assignments) != len(slot_by_id):
    return None
  ordered_slots = sorted(target_slots, key=lambda slot: (
      int(slot.get("layer_index", 0)), target_slots.index(slot)))
  return {
      "placements": [{"object_id": assignments[str(slot["slot_id"])],
                      "slot_id": str(slot["slot_id"])}
                     for slot in ordered_slots],
      "summary": "",
  }
