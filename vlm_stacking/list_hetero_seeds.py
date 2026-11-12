"""Inspect hetero-pyramid-data-test and list seeds by object diversity.

Reads the dataset info pickles directly — no PyBullet or TensorFlow needed.

Usage (from repo root):
  python -m vlm_stacking.list_hetero_seeds
  python -m vlm_stacking.list_hetero_seeds --dataset data/hetero-pyramid-data/hetero-pyramid-data-test
  python -m vlm_stacking.list_hetero_seeds --min-shapes 3
  python -m vlm_stacking.list_hetero_seeds --layers 2

To demo a seed found here, pass it to demo.py via the dataset:
  python -m vlm_stacking.demo \\
      --seed-dataset data/hetero-pyramid-data/hetero-pyramid-data-test \\
      --seed-index <N>

NOTE: demo.py runs the stack-block-pyramid-parallel-jaw task, which always
spawns same-size cubes regardless of seed. To see heterogeneous shapes in the
simulator, the task must be changed to one that spawns mixed objects (e.g.
mixed-shape-stacking). The seeds here are useful for reproducing specific
random initial placements.
"""

import argparse
import glob
import os
import pickle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(
    REPO_ROOT, "data", "hetero-pyramid-data", "hetero-pyramid-data-test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Path to hetero-pyramid dataset folder (default: %(default)s)")
    parser.add_argument("--min-shapes", type=int, default=1,
                        help="Only show episodes with at least this many distinct shapes")
    parser.add_argument("--layers", type=int, default=None,
                        help="Only show episodes with this many layers")
    parser.add_argument("--sort", choices=("diversity", "seed", "objects"),
                        default="diversity",
                        help="Sort order: diversity (default), seed, or objects")
    return parser.parse_args()


def load_episodes(dataset_path):
    info_dir = os.path.join(dataset_path, "info")
    if not os.path.isdir(info_dir):
        raise FileNotFoundError("Dataset info folder not found: %s" % info_dir)
    files = sorted(glob.glob(os.path.join(info_dir, "*.pkl")))
    if not files:
        raise ValueError("No .pkl files in %s" % info_dir)

    episodes = []
    for filepath in files:
        fname = os.path.basename(filepath)
        ep_idx = int(fname[:6])
        seed = int(fname[fname.find("-") + 1:-4])
        with open(filepath, "rb") as fh:
            info = pickle.load(fh)

        # Find first step that has an objects list.
        step0 = next((s for s in info if isinstance(s, dict) and "objects" in s), None)
        if step0 is None:
            continue

        objects = step0["objects"]
        shapes = [o["shape"] for o in objects]
        distinct_shapes = sorted(set(shapes))
        num_layers = step0.get("num_layers", "?")
        stack_type = step0.get("stack_type", "?")

        # Shape counts for display.
        shape_counts = {}
        for s in shapes:
            shape_counts[s] = shape_counts.get(s, 0) + 1
        shape_summary = "+".join(
            "%dx%s" % (shape_counts[s], s) for s in distinct_shapes)

        episodes.append({
            "ep_idx": ep_idx,
            "seed": seed,
            "num_layers": num_layers,
            "n_objects": len(objects),
            "distinct_shapes": distinct_shapes,
            "n_distinct": len(distinct_shapes),
            "shape_summary": shape_summary,
            "stack_type": stack_type,
        })

    return episodes


def main():
    args = parse_args()

    if not os.path.isdir(args.dataset):
        raise SystemExit("Dataset not found: %s" % args.dataset)

    episodes = load_episodes(args.dataset)

    # Filter.
    filtered = [e for e in episodes if e["n_distinct"] >= args.min_shapes]
    if args.layers is not None:
        filtered = [e for e in filtered if e["num_layers"] == args.layers]

    if not filtered:
        print("No episodes match the filter criteria.")
        return

    # Sort.
    if args.sort == "diversity":
        filtered.sort(key=lambda e: (-e["n_distinct"], -e["n_objects"], e["seed"]))
    elif args.sort == "seed":
        filtered.sort(key=lambda e: e["seed"])
    elif args.sort == "objects":
        filtered.sort(key=lambda e: (-e["n_objects"], -e["n_distinct"], e["seed"]))

    # Print table.
    print("Dataset: %s" % os.path.abspath(args.dataset))
    print("Total episodes: %d  |  Shown after filter: %d\n" % (len(episodes), len(filtered)))
    header = "%-6s  %-5s  %-6s  %-6s  %-9s  %s" % (
        "ep_idx", "seed", "layers", "n_obj", "n_shapes", "composition")
    print(header)
    print("-" * len(header))
    for e in filtered:
        print("%-6d  %-5d  %-6s  %-6d  %-9d  %s" % (
            e["ep_idx"], e["seed"], e["num_layers"],
            e["n_objects"], e["n_distinct"], e["shape_summary"]))

    # Summary by diversity tier.
    print()
    by_distinct = {}
    for e in episodes:
        key = e["n_distinct"]
        by_distinct.setdefault(key, []).append(e["seed"])
    print("Diversity summary (all episodes):")
    for k in sorted(by_distinct.keys(), reverse=True):
        seeds = sorted(by_distinct[k])
        print("  %d distinct shape(s): %d episode(s), seeds: %s" % (
            k, len(seeds), seeds))

    # Top recommendations.
    top = [e for e in filtered if e["n_distinct"] == max(e2["n_distinct"] for e2 in filtered)]
    print()
    print("Most heterogeneous seeds (use with --seed-dataset and --seed-index):")
    for e in top[:10]:
        print("  seed %-4d  (ep_idx=%-3d  layers=%-2s  objects=%-2d  shapes=%s)" % (
            e["seed"], e["ep_idx"], e["num_layers"], e["n_objects"], e["shape_summary"]))
    print()
    print("Example demo command (edit task name if needed):")
    best = top[0]
    print("  python -m vlm_stacking.demo \\")
    print("      --seed-dataset %s \\" % os.path.abspath(args.dataset))
    print("      --seed-index %d" % best["ep_idx"])


if __name__ == "__main__":
    main()
