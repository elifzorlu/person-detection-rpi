import os
import random
import argparse
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg")
CATEGORIES = ["person", "non_person"]

def list_images(data_dir: Path, category: str):
    """Return list of relative paths like 'person/xxx.jpg'."""
    cat_dir = data_dir / category
    if not cat_dir.exists():
        raise FileNotFoundError(f"Missing folder: {cat_dir}")
    files = []
    for f in sorted(cat_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            files.append(f"{category}/{f.name}")
    return files

def write_manifest(out_path: Path, relpaths):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in relpaths:
            f.write(p + "\n")

def make_split(files, val_frac, test_frac, seed):
    """
    Deterministically split a list into train/val/test.
    Uses a per-class shuffle with the provided seed.
    """
    rng = random.Random(seed)
    files = files[:]  # copy
    rng.shuffle(files)

    n = len(files)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test = files[:n_test]
    val = files[n_test:n_test + n_val]
    train = files[n_test + n_val:]
    return train, val, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="vw_coco2014_96",
                    help="Dataset root containing person/ and non_person/")
    ap.add_argument("--out", type=str, default="splits",
                    help="Output folder for manifests")
    ap.add_argument("--val", type=float, default=0.1,
                    help="Validation fraction (default 0.1)")
    ap.add_argument("--test_public", type=float, default=0.1,
                    help="Public test fraction (default 0.1)")
    ap.add_argument("--seed", type=int, default=341,
                    help="Base seed for deterministic splits")
    ap.add_argument("--hidden_seed", type=int, default=99991,
                    help="Seed for hidden test manifest generation")
    ap.add_argument("--hidden_size", type=int, default=0,
                    help="Number of images per class in hidden test. 0 = same count as public test")
    ap.add_argument("--write_hidden", action="store_true",
                    help="Generate test_hidden.txt (instructor only, not for students)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_dir.resolve()}")

    # 1) Public split (train/val/test_public), stratified per class
    train_all, val_all, test_public_all = [], [], []

    for cat in CATEGORIES:
        files = list_images(data_dir, cat)
        tr, va, te = make_split(files, args.val, args.test_public, seed=args.seed)
        train_all += tr
        val_all += va
        test_public_all += te

    # Shuffle each final list (so not grouped by class)
    rng_pub = random.Random(args.seed)
    rng_pub.shuffle(train_all)
    rng_pub.shuffle(val_all)
    rng_pub.shuffle(test_public_all)

    write_manifest(out_dir / "train.txt", train_all)
    write_manifest(out_dir / "val.txt", val_all)
    write_manifest(out_dir / "test_public.txt", test_public_all)

    # 2) Hidden test manifest (instructor only - requires --write_hidden flag)
    if args.write_hidden:
        # Ensure hidden test is disjoint from ALL public splits (train/val/test_public)
        exclude_set = set(train_all) | set(val_all) | set(test_public_all)

        test_hidden_all = []
        for cat in CATEGORIES:
            files = list_images(data_dir, cat)

            # Exclude all public split items to ensure disjointness
            files = [f for f in files if f not in exclude_set]

            rng_hid = random.Random(args.hidden_seed + (0 if cat == "person" else 1))
            rng_hid.shuffle(files)

            if args.hidden_size and args.hidden_size > 0:
                k = min(args.hidden_size, len(files))
            else:
                # match per-class count of public test
                public_count_cat = sum(1 for x in test_public_all if x.startswith(cat + "/"))
                k = min(public_count_cat, len(files))

            test_hidden_all += files[:k]

        rng_hfinal = random.Random(args.hidden_seed)
        rng_hfinal.shuffle(test_hidden_all)
        write_manifest(out_dir / "test_hidden.txt", test_hidden_all)
        
        # Sanity check: ensure disjointness
        assert not (set(test_hidden_all) & exclude_set), "Hidden test overlaps with public splits!"
    else:
        test_hidden_all = None
    
    # Summary
    def count_cat(items, cat):
        return sum(1 for x in items if x.startswith(cat + "/"))

    print("========================================")
    print("[SUCCESS] Wrote deterministic manifests:")
    print(f"  {str((out_dir / 'train.txt').resolve())}")
    print(f"  {str((out_dir / 'val.txt').resolve())}")
    print(f"  {str((out_dir / 'test_public.txt').resolve())}")
    if args.write_hidden:
        print(f"  {str((out_dir / 'test_hidden.txt').resolve())}")
    print("----------------------------------------")
    print("Public split (seed = {}):".format(args.seed))
    print(f"  train:       {len(train_all)}  (person={count_cat(train_all,'person')}, non_person={count_cat(train_all,'non_person')})")
    print(f"  val:         {len(val_all)}    (person={count_cat(val_all,'person')}, non_person={count_cat(val_all,'non_person')})")
    print(f"  test_public: {len(test_public_all)} (person={count_cat(test_public_all,'person')}, non_person={count_cat(test_public_all,'non_person')})")
    if args.write_hidden:
        print("Hidden test (seed = {}, disjoint from ALL public splits):".format(args.hidden_seed))
        print(f"  test_hidden: {len(test_hidden_all)} (person={count_cat(test_hidden_all,'person')}, non_person={count_cat(test_hidden_all,'non_person')})")
    else:
        print("Hidden test: NOT generated (use --write_hidden flag, instructor only)")
    print("========================================")

if __name__ == "__main__":
    main()