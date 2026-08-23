import glob
import os
import random


def deduplicate_rmb_files(filenames, clean_suffix="_clean"):
    """Collapse a raw/cleaned pair of the same episode (e.g. "X.rmb" and
    "X_clean.rmb", the naming misc/FilterUmiTrackerSpikes.py's
    --output_suffix example uses) down to a single file, preferring the
    cleaned version. Without this, a directory containing both counts as two
    independent episodes: it inflates the apparent dataset size and, worse,
    lets the same underlying recording land on both sides of a train/val
    split (the "held-out" file is then a near-duplicate of something the
    model already trained on, so the val loss no longer measures real
    generalization)."""

    def dedup_key(path):
        root, ext = os.path.splitext(path)
        if root.endswith(clean_suffix):
            root = root[: -len(clean_suffix)]
        return root + ext

    best_by_base = {}
    for path in filenames:
        base = dedup_key(path)
        is_clean = base != path
        if base not in best_by_base or (is_clean and not best_by_base[base][1]):
            best_by_base[base] = (path, is_clean)

    return sorted(path for path, _ in best_by_base.values())


def find_rmb_files(base_path, num_files=None, dedupe=False):
    if base_path.rstrip("/").endswith((".rmb", ".hdf5")):
        rmb_path_list = [base_path]
    elif os.path.isdir(base_path):
        rmb_path_list = sorted(
            [
                f
                for f in glob.glob(f"{base_path}/**/*.*", recursive=True)
                if f.endswith(".rmb")
                or (f.endswith(".hdf5") and not f.endswith(".rmb.hdf5"))
            ]
        )
    else:
        raise ValueError(f"[find_rmb_files] RMB file not found: {base_path}")

    if dedupe:
        rmb_path_list = deduplicate_rmb_files(rmb_path_list)

    if num_files is not None:
        if num_files > len(rmb_path_list):
            raise ValueError(
                f"[find_rmb_files] Requested num_files={num_files} exceeds total available files={len(rmb_path_list)}."
            )
        rmb_path_list = sorted(random.sample(rmb_path_list, num_files))

    return rmb_path_list
