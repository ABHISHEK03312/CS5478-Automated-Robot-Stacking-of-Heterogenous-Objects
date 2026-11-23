"""Stack objects into two equal towers, using VLM assignment + CEM placement refinement.

The VLM assigns objects to two towers (A and B, N/2 objects each). CEM then
refines the continuous (x, y, yaw) for each placement around the nominal slot
position. The GUI replay uses the CEM-refined poses via direct teleport so the
video does not re-run the search.

Supports any even number of objects in the scene (6 → 2×3 towers,
10 → 2×5 towers, etc.).

Usage:
    python -m vlm_stacking.demo_two_stacks --seed-index 42

Record a video:
    python -m vlm_stacking.demo_two_stacks --seed-index 42 --video two_stacks.mp4
"""

import argparse
import json
import os
import time

import numpy as np
import pybullet as p

from ravens import tasks
from ravens.environments.environment import Environment

from vlm_stacking import evaluate
from vlm_stacking.hetero_task import load_hetero_episode_info, make_hetero_episode_task
from vlm_stacking.planner import OllamaStackPlanner
from vlm_stacking.ravens_sim import RavensSimulationSceneProvider

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_NAME = "stack-block-pyramid-parallel-jaw"
DEFAULT_ASSETS_ROOT = os.path.join(REPO_ROOT, "ravens", "environments", "assets")
DEFAULT_SEED_DATASET = os.path.join(
    REPO_ROOT, "data", "parallel-jaw-v2", TASK_NAME + "-test")

# Two tower XY positions within workspace bounds [0.25,0.75] × [-0.50,0.50]
TOWER_A_XY = (0.40, 0.18)
TOWER_B_XY = (0.60, 0.18)
FIRST_LAYER_Z = 0.03   # Z-center of the bottom object in each tower
LAYER_HEIGHT  = 0.055  # Z increment per stacking layer


# ---------------------------------------------------------------------------
# Dynamic two-tower slot layout
# ---------------------------------------------------------------------------

def make_two_tower_slots(n_objects):
    """Return a dict of n_objects slots: n//2 layers per tower (A and B)."""
    n_layers = n_objects // 2
    slots = {}
    for tower_label, (tx, ty) in [("a", TOWER_A_XY), ("b", TOWER_B_XY)]:
        for layer in range(n_layers):
            slot_id = "tower_%s_layer_%d" % (tower_label, layer)
            z = FIRST_LAYER_Z + layer * LAYER_HEIGHT
            slots[slot_id] = {
                "slot_id": slot_id,
                "tower": tower_label,
                "layer_index": layer,
                "pose": ((tx, ty, z), (0.0, 0.0, 0.0, 1.0)),
                "description": "Layer %d of tower %s" % (
                    layer, tower_label.upper()),
            }
    return slots


# ---------------------------------------------------------------------------
# Custom VLM prompt (no pyramid constraint)
# ---------------------------------------------------------------------------

def build_two_tower_prompt(instruction, objects_for_prompt, slots):
    n_objects = len(objects_for_prompt)
    n_layers  = n_objects // 2
    slot_catalog = [
        {
            "slot_id": s["slot_id"],
            "tower": s["tower"].upper(),
            "layer": s["layer_index"],
            "description": s["description"],
        }
        for s in sorted(slots.values(),
                        key=lambda x: (x["tower"], x["layer_index"]))
    ]
    return (
        "Assign exactly %(n)d objects to two towers of %(k)d objects each. "
        "Tower A slots: tower_a_layer_0 (bottom) through tower_a_layer_%(km1)d (top). "
        "Tower B slots: tower_b_layer_0 (bottom) through tower_b_layer_%(km1)d (top). "
        "Each object must be assigned to exactly one slot. "
        "Each slot must be used exactly once. "
        "Within each tower, place layer_0 first, then layer_1, and so on upward. "
        "To build both towers simultaneously, interleave towers by layer: "
        "place one object from layer_0 for tower A, then one from layer_0 for "
        "tower B, then layer_1 for tower A, then layer_1 for tower B, and so on. "
        "Put larger or heavier objects in lower layers for stability. "
        "Do NOT use pyramid slot names. Do NOT return coordinates. "
        "User instruction: %(inst)s\n\n"
        "Measured object catalog (%(n)d objects):\n%(catalog)s\n\n"
        "Available slots (%(n)d total):\n%(slots)s\n\n"
        "Return ONLY JSON: {\"placements\":[{\"object_id\":\"...\","
        "\"slot_id\":\"...\"}],\"summary\":\"...\"}" % {
            "n":     n_objects,
            "k":     n_layers,
            "km1":   n_layers - 1,
            "inst":  instruction,
            "catalog": json.dumps(objects_for_prompt, sort_keys=True),
            "slots":   json.dumps(slot_catalog, sort_keys=True),
        }
    )


# ---------------------------------------------------------------------------
# Custom two-tower validator
# ---------------------------------------------------------------------------

def validate_two_tower_plan(plan, objects_for_prompt, slots):
    """Check full coverage and ascending layer order within each tower."""
    if not isinstance(plan, dict) or not isinstance(plan.get("placements"), list):
        raise ValueError("Plan must have a 'placements' list.")
    placements = plan["placements"]
    expected_object_ids = {str(o["object_id"]) for o in objects_for_prompt}
    expected_slot_ids   = set(slots.keys())

    used_objects, used_slots = set(), set()
    for entry in placements:
        oid = str(entry.get("object_id", ""))
        sid = str(entry.get("slot_id", ""))
        if oid not in expected_object_ids:
            raise ValueError("Unknown object_id: %r" % oid)
        if sid not in expected_slot_ids:
            raise ValueError(
                "Unknown slot_id %r. Valid: %s" % (sid, sorted(expected_slot_ids)))
        if oid in used_objects:
            raise ValueError("Object %s assigned twice." % oid)
        if sid in used_slots:
            raise ValueError("Slot %s used twice." % sid)
        used_objects.add(oid)
        used_slots.add(sid)

    if used_objects != expected_object_ids:
        raise ValueError("Objects not assigned: %s" % sorted(
            expected_object_ids - used_objects))
    if used_slots != expected_slot_ids:
        raise ValueError("Slots not filled: %s" % sorted(
            expected_slot_ids - used_slots))

    # Within each tower, layers must appear in ascending order in the plan.
    for tower_label in ("a", "b"):
        prefix = "tower_%s_layer_" % tower_label
        seq = sorted(
            [(idx, slots[str(e["slot_id"])]["layer_index"])
             for idx, e in enumerate(placements)
             if str(e.get("slot_id", "")).startswith(prefix)],
            key=lambda x: x[0])
        layers_in_order = [li for _, li in seq]
        if layers_in_order != sorted(layers_in_order):
            raise ValueError(
                "Tower %s layers not ascending: %s" % (
                    tower_label.upper(), layers_in_order))
    return plan


# ---------------------------------------------------------------------------
# CEM pose refinement
# ---------------------------------------------------------------------------

def cem_refine_pose(object_id, nominal_pos, nominal_quat,
                    cem_samples=24, cem_elites=5, cem_iters=3,
                    sigma_xy=0.012, sigma_yaw=0.3, rng=None):
    """Refine (pos, quat) around nominal using CEM teleport-evaluate loop.

    Runs entirely in the currently active PyBullet simulation. The object is
    teleported to candidate poses, physics is stepped briefly, and the XY drift
    (stability proxy) is measured. The object is restored after each candidate.
    Returns the (pos, quat) of the elite candidate with lowest XY drift.
    """
    if rng is None:
        rng = np.random.default_rng()
    x0, y0, z0 = float(nominal_pos[0]), float(nominal_pos[1]), float(nominal_pos[2])
    save_pos, save_orn = p.getBasePositionAndOrientation(object_id)

    mean = np.array([0.0, 0.0, 0.0])  # [delta_x, delta_y, delta_yaw]
    std  = np.array([sigma_xy, sigma_xy, sigma_yaw])

    best_pos  = (x0, y0, z0)
    best_quat = nominal_quat
    best_score = -np.inf

    try:
        for _ in range(cem_iters):
            samples = rng.normal(mean, std, size=(cem_samples, 3))
            scores  = np.zeros(cem_samples)

            for i, (dx, dy, dyaw) in enumerate(samples):
                cand_pos  = (x0 + float(dx), y0 + float(dy), z0)
                cand_quat = p.getQuaternionFromEuler([0.0, 0.0, float(dyaw)])
                p.resetBasePositionAndOrientation(object_id, cand_pos, cand_quat)
                for _ in range(15):
                    p.stepSimulation()
                final_pos, _ = p.getBasePositionAndOrientation(object_id)
                # Score: negative XY drift from candidate (lower drift = more stable)
                drift = float(np.linalg.norm(
                    np.array(final_pos[:2]) - np.array(cand_pos[:2])))
                scores[i] = -drift
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_pos   = cand_pos
                    best_quat  = cand_quat

                # Restore for next candidate
                p.resetBasePositionAndOrientation(object_id, save_pos, save_orn)

            # Elite distribution update
            elite_idx    = np.argsort(scores)[-cem_elites:]
            elite_samples = samples[elite_idx]
            mean = elite_samples.mean(axis=0)
            std  = elite_samples.std(axis=0) + 1e-4

    finally:
        p.resetBasePositionAndOrientation(object_id, save_pos, save_orn)

    return best_pos, best_quat


# ---------------------------------------------------------------------------
# Headless: run CEM per placement, execute sequentially, collect refined poses
# ---------------------------------------------------------------------------

def run_cem_headless(env, plan, slots, cem_samples, cem_elites, cem_iters,
                     search_seed, pause=0.0):
    """Execute plan in headless env with CEM refinement; return refined poses.

    Objects are placed one by one so each CEM call sees the actual stack state
    (layer N CEM evaluates on top of the already-placed layer N-1 objects).
    """
    rng = np.random.default_rng(search_seed)
    refined_poses = {}  # slot_id -> (pos, quat)

    for entry in plan["placements"]:
        object_id = int(entry["object_id"])
        slot      = slots[entry["slot_id"]]
        nominal_pos, nominal_quat = slot["pose"]

        best_pos, best_quat = cem_refine_pose(
            object_id, nominal_pos, nominal_quat,
            cem_samples=cem_samples, cem_elites=cem_elites,
            cem_iters=cem_iters, rng=rng)

        refined_poses[entry["slot_id"]] = (best_pos, best_quat)

        # Execute the refined placement so the next CEM call sees correct stack.
        pick_pos, pick_orn = p.getBasePositionAndOrientation(object_id)
        action = {
            "pose0": (pick_pos, pick_orn),
            "pose1": (best_pos, best_quat),
        }
        env.step(action)
        if pause > 0:
            time.sleep(pause)

    return refined_poses


# ---------------------------------------------------------------------------
# GUI replay via teleport (same pattern as demo.py _teleport_replay)
# ---------------------------------------------------------------------------

def teleport_replay(gui_env, gui_task, scene_seed, plan, refined_poses):
    """Reset scene then teleport each object to its CEM-refined target pose."""
    np.random.seed(scene_seed)
    gui_env.seed(scene_seed)
    gui_env.set_task(gui_task)
    gui_env.reset()
    for entry in plan.get("placements", []):
        object_id = int(entry["object_id"])
        sid = entry["slot_id"]
        if sid not in refined_poses:
            continue
        place_pos, place_quat = refined_poses[sid]
        pick_pos, pick_orn = p.getBasePositionAndOrientation(object_id)
        action = {
            "pose0": (pick_pos, pick_orn),
            "pose1": (place_pos, place_quat),
        }
        _, _, done, _ = gui_env.step(action)
        if done:
            break


# ---------------------------------------------------------------------------
# CLI — mirrors demo.py argument layout
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # Seed / dataset — identical to demo.py
    parser.add_argument("--seed", type=int, default=None,
                        help="Exact PyBullet scene seed; if omitted, derived "
                             "from --seed-dataset at --seed-index")
    parser.add_argument("--seed-dataset", default=DEFAULT_SEED_DATASET)
    parser.add_argument("--seed-index", type=int, default=42,
                        help="0-based episode index within --seed-dataset "
                             "(default: 42)")
    parser.add_argument("--seed-offset", type=int, default=0)
    # VLM
    parser.add_argument("--model", default=None,
                        help="Ollama VLM model (default: OLLAMA_VLM_MODEL)")
    parser.add_argument("--temperature", type=float, default=0.0001)
    parser.add_argument("--instruction",
                        default="Stack all objects into two equal, stable towers.")
    # CEM
    parser.add_argument("--search-seed", type=int, default=11)
    parser.add_argument("--cem-samples", type=int, default=24)
    parser.add_argument("--cem-elites", type=int, default=5)
    parser.add_argument("--cem-iterations", type=int, default=3)
    # Paths
    parser.add_argument("--assets-root", default=DEFAULT_ASSETS_ROOT)
    # Replay
    parser.add_argument("--video-teleport", action="store_true", default=True,
                        help="GUI replay teleports objects to CEM-refined poses "
                             "(default: True; pass --no-video-teleport to "
                             "re-execute CEM live in GUI)")
    parser.add_argument("--no-video-teleport", dest="video_teleport",
                        action="store_false")
    parser.add_argument("--video", default=None,
                        help="MP4 output path "
                             "(default: vlm_two_stacks_seed_<N>.mp4)")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Continue to GUI even if headless CEM produced "
                             "low reward")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Seed resolution (mirrors demo.py) ---
    if args.seed is not None:
        scene_seed = args.seed
    else:
        seeds = evaluate.dataset_seeds(
            args.seed_dataset, seed_offset=args.seed_offset)
        if args.seed_index >= len(seeds):
            raise SystemExit(
                "--seed-index %d out of range: dataset has %d episodes." % (
                    args.seed_index, len(seeds)))
        scene_seed = seeds[args.seed_index]
        print("Scene seed: %d  (dataset index %d of %d)" % (
            scene_seed, args.seed_index, len(seeds)))

    # --- Hetero episode detection (mirrors demo.py) ---
    hetero_info = None
    if args.seed is None:
        try:
            candidate_info, _ = load_hetero_episode_info(
                args.seed_dataset, args.seed_index)
            step0 = next(
                (s for s in candidate_info
                 if isinstance(s, dict) and "objects" in s and
                 s["objects"] and "shape" in s["objects"][0]),
                None)
            if step0 is not None:
                hetero_info = candidate_info
                print("Hetero episode detected: %d object(s), shapes=%s" % (
                    len(step0["objects"]),
                    sorted({o["shape"] for o in step0["objects"]})))
        except Exception:
            pass

    def make_task():
        if hetero_info is not None:
            t = make_hetero_episode_task(hetero_info)()
        else:
            t = tasks.names[TASK_NAME]()
        t.mode = "test"
        return t

    # --- Headless pass: get catalog, plan with VLM, refine with CEM ---
    task = make_task()

    headless_env = Environment(args.assets_root, disp=False, hz=480)
    plan = None
    refined_poses = {}
    try:
        np.random.seed(scene_seed)
        headless_env.seed(scene_seed)
        headless_env.set_task(task)
        headless_env.reset()

        scene_provider = RavensSimulationSceneProvider(headless_env, task)
        image_bytes, objects, objects_for_prompt, _ = scene_provider()

        n = len(objects_for_prompt)
        print("Objects in scene: %d" % n)
        if n < 2:
            raise SystemExit("Need at least 2 objects; got %d." % n)
        if n % 2 != 0:
            print("Warning: odd object count (%d); one tower will have one "
                  "extra object." % n)

        slots = make_two_tower_slots(n)

        planner = OllamaStackPlanner(model=args.model, temperature=args.temperature)
        prompt  = build_two_tower_prompt(args.instruction, objects_for_prompt, slots)

        last_error = None
        for attempt in range(1, 4):
            try:
                raw  = planner._generate_plan(prompt, image_bytes)
                plan = validate_two_tower_plan(raw, objects_for_prompt, slots)
                print("VLM plan validated on attempt %d." % attempt)
                break
            except Exception as err:
                last_error = err
                print("Attempt %d failed: %s" % (attempt, err))
        if plan is None:
            raise SystemExit(
                "VLM could not produce a valid two-tower plan after 3 attempts.\n"
                "Last error: %s" % last_error)

        print("\nVLM assignment:")
        for entry in plan["placements"]:
            print("  object %-4s  →  %s" % (
                entry["object_id"], entry["slot_id"]))

        print("\nRunning CEM refinement (%d samples × %d iters)..." % (
            args.cem_samples, args.cem_iterations))
        refined_poses = run_cem_headless(
            headless_env, plan, slots,
            cem_samples=args.cem_samples,
            cem_elites=args.cem_elites,
            cem_iters=args.cem_iterations,
            search_seed=args.search_seed)

        print("CEM-refined placements:")
        for sid, (pos, _) in refined_poses.items():
            print("  %-24s  pos=(%.3f, %.3f, %.3f)" % (
                sid, pos[0], pos[1], pos[2]))

    finally:
        headless_env.close()

    # --- GUI replay with video recording ---
    gui_task = make_task()
    gui_env = Environment(args.assets_root, task=gui_task, disp=True, hz=480)
    try:
        video_path = args.video or "vlm_two_stacks_seed_%d.mp4" % scene_seed
        video_path = os.path.abspath(video_path)
        video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
        print("\nRecording GUI to %s" % video_path)

        if args.video_teleport:
            print("Replaying via teleport (CEM-refined poses).")
            teleport_replay(gui_env, gui_task, scene_seed, plan, refined_poses)
        else:
            # Re-run CEM live in GUI (slow but shows search behaviour)
            np.random.seed(scene_seed)
            gui_env.seed(scene_seed)
            gui_env.set_task(gui_task)
            gui_env.reset()
            run_cem_headless(
                gui_env, plan, slots,
                cem_samples=args.cem_samples,
                cem_elites=args.cem_elites,
                cem_iters=args.cem_iterations,
                search_seed=args.search_seed,
                pause=0.3)

        p.stopStateLogging(video_log_id)
        print("Video saved to %s" % video_path)

        if not args.no_pause:
            input("\nDone. Press Enter to close PyBullet. ")
    finally:
        gui_env.close()


if __name__ == "__main__":
    main()
