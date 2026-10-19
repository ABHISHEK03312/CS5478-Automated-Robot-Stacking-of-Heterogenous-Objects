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
        "stack_order": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Object IDs ordered from bottom of stack to top.",
        },
        "summary": {"type": "string"},
    },
    "required": ["stack_order", "summary"],
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

  def plan(self, instruction, image_bytes, objects):
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

    content_type = "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
      content_type = "image/jpeg"
    image_url = "data:%s;base64,%s" % (
        content_type, base64.b64encode(image_bytes).decode("ascii"))

    prompt = (
        "Plan a single stable stack that follows the user's instruction. "
        "Use only the supplied object IDs. Return every pickable object exactly "
        "once in stack_order, from bottom to top. Prefer centered support and "
        "place larger, flatter objects below smaller objects when that is "
        "consistent with the instruction. Do not invent dimensions, object IDs, "
        "or physical properties. If the instruction conflicts with the catalog, "
        "still return the closest physically plausible order; the robot-side "
        "validator will reject an unsafe order.\n\n"
        "User instruction: %s\n\n"
        "Measured object catalog (dimensions are meters):\n%s" % (
            instruction, json.dumps(objects, sort_keys=True)))
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
    validate_plan(plan, objects)
    return plan

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

    def plan(self, instruction, image_bytes, objects):
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

        previous_output = None
        repair_error = None
        repair_kind = None
        for attempt in range(self.max_retries + 1):
            prompt = self._build_prompt(
                instruction, objects, previous_output, repair_error,
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
                validate_plan(plan, objects)
            except StackPlanError as error:
                previous_output = json.dumps(plan, ensure_ascii=False)
                repair_error = str(error)
                repair_kind = "plan validation"
                if attempt >= self.max_retries:
                    raise StackPlanError(
                        "Could not produce a valid plan after %d retries "
                        "(%d attempts). Last validation error: %s" % (
                            self.max_retries, self.max_retries + 1, error))
                continue
            return plan

    @staticmethod
    def _build_prompt(instruction, objects, previous_output=None,
                      repair_error=None, repair_kind=None):
        # prompt = (
        #     "Plan a single stable stack that follows the user's instruction. "
        #     "Use only the supplied object IDs. Return every pickable object "
        #     "exactly once in stack_order, from bottom to top. Prefer centered "
        #     "support and place larger, flatter objects below smaller objects "
        #     "when that is consistent with the instruction. Do not invent "
        #     "dimensions, object IDs, or physical properties. If the instruction "
        #     "conflicts with the catalog, still return the closest physically "
        #     "plausible order.\n\n"
        #     "User instruction: %s\n\n"
        #     "Measured object catalog (dimensions are meters):\n%s\n\n"
        #     "Respond with ONLY a JSON object in this exact format, no "
        #     "explanation, no markdown fences:\n"
        #     '{"stack_order": ["<bottom_object_id>", ..., "<top_object_id>"], '
        #     '"summary": "<brief explanation>"}'
        #     % (instruction, json.dumps(objects, sort_keys=True)))
        prompt = (
              "Plan a single stable stack that follows the user's instruction. "
              "Use only the supplied object IDs. Return every pickable object "
              "exactly once in stack_order, from bottom to top. Prefer centered "
              "support and place larger, flatter objects below smaller objects "
              "when that is consistent with the instruction. Do not invent "
              "dimensions, object IDs, or physical properties. If the instruction "
              "conflicts with the catalog, still return the closest physically "
              "plausible order.\n\n"
              "User instruction: %s\n\n"
              "Measured object catalog (dimensions are meters):\n%s\n\n"
              "Respond with ONLY a JSON object in this exact format, no "
              "explanation, no markdown fences:\n"
              '{"stack_order": ["<bottom_object_id>", ..., "<top_object_id>"], '
              '"summary": "<brief explanation>"}'
              % (instruction, json.dumps(objects, sort_keys=True)))
        if previous_output is not None or repair_error is not None:
            required_ids = [str(obj["object_id"]) for obj in objects]
            prompt += (
                "\n\nYour previous response failed %s checks. Correct "
                "the response using the error below. The required object IDs "
                "are %s; include every one exactly once in stack_order, and "
                "do not include any other IDs. Fix any unsafe footprint order "
                "identified by the validator. Return only the complete JSON "
                "object.\nPrevious response or plan:\n%s\nError:\n%s\n" % (
                    repair_kind or "plan",
                    json.dumps(required_ids),
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
            text = self._extract_json(text)
            return json.loads(text)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raw_output = (response.get("message", {}).get("content", "")
                          if isinstance(response, dict) else repr(response))
            raise StackPlanError(
                "VLM response was not a parseable plan JSON object: %s\n"
                "Raw output: %s" % (error, raw_output))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text):
        """Strip markdown code fences and leading prose to isolate JSON."""
        # Remove ```json ... ``` or ``` ... ``` wrappers.
        stripped = text.strip()
        if stripped.startswith("```"):
            # Drop the opening fence (with optional language tag).
            first_newline = stripped.index("\n")
            stripped = stripped[first_newline + 1 :]
            # Drop the closing fence.
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3]
            return stripped.strip()

        # If the model prefixed prose before the JSON object, try to find it.
        brace = stripped.find("{")
        if brace != -1:
            # Find the matching closing brace (outermost).
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


def validate_plan(plan, objects, footprint_tolerance=0.005):
  """Validate IDs and simple footprint support before robot execution.

  Object dimensions must be measured by the scene adapter. This deliberately
  checks only a conservative geometric condition; it is not a stability proof.
  """
  if not isinstance(plan, dict) or not isinstance(plan.get("stack_order"), list):
    raise StackPlanError("Plan must contain a stack_order list.")
  ids = [str(obj["object_id"]) for obj in objects]
  order = [str(obj_id) for obj_id in plan["stack_order"]]
  if len(order) != len(set(order)):
    raise StackPlanError("Plan repeats an object ID: %s" % order)
  if set(order) != set(ids):
    raise StackPlanError(
        "Plan object IDs must match the visible object catalog exactly. Output IDs %s, expected IDs %s" % (order, ids))
  if not isinstance(plan.get("summary", ""), str):
    raise StackPlanError("Plan summary must be text.")

  by_id = {str(obj["object_id"]): obj for obj in objects}
  for lower_id, upper_id in zip(order, order[1:]):
    lower = by_id[lower_id].get("dimensions")
    upper = by_id[upper_id].get("dimensions")
    if lower is None or upper is None or len(lower) < 2 or len(upper) < 2:
      continue
    # Compare the two horizontal footprint axes without assuming which camera
    # axis is object width. Conservative sorting handles rectangular objects.
    lower_xy = sorted([float(lower[0]), float(lower[1])])
    upper_xy = sorted([float(upper[0]), float(upper[1])])
    if any(u > l + footprint_tolerance
           for u, l in zip(upper_xy, lower_xy)):
      raise StackPlanError(
          "Unsafe footprint order: %s cannot fully support %s." % (
              lower_id, upper_id))
  return plan
