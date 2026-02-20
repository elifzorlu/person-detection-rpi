"""Raspberry Pi Scoreboard and Profiler for VWW Models.

This script evaluates TFLite models on Raspberry Pi with:
- Manifest-based test splits (from create_main_datasplit.py)
- Warmup phase for stable measurements
- Detailed latency percentiles (p50/p90/p99)
- Memory profiling
- Competition score calculation
- Device information capture

Usage:
    python src/scoreboard.py --model models/vww_96.tflite --split test_public
"""

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import psutil
except ImportError:
    psutil = None
    print("[WARN] psutil not found. Memory tracking will be limited.")

import resource

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    raise SystemExit("[ERROR] tflite_runtime not found. Install: pip install tflite-runtime")

# Default paths
BASE_DIR = "vw_coco2014_96"
SPLITS_DIR = "splits"

# Class mapping
CLASS_MAP = {"non_person": 0, "person": 1}


def read_device_info():
    """Capture device information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    
    # Raspberry Pi model string (if present)
    model_path = "/proc/device-tree/model"
    if os.path.exists(model_path):
        try:
            with open(model_path, "r") as f:
                info["rpi_model"] = f.read().strip("\x00").strip()
        except Exception:
            pass

    # CPU info (best-effort)
    cpuinfo = "/proc/cpuinfo"
    if os.path.exists(cpuinfo):
        try:
            with open(cpuinfo, "r") as f:
                text = f.read()
            for line in text.splitlines():
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    
    # CPU governor
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
            info["cpu_governor"] = f.read().strip()
    except:
        pass
    
    return info


def load_manifest(manifest_path):
    """Load image paths from manifest file."""
    with open(manifest_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_and_preprocess_image(image_path: str, input_hw):
    """Load and preprocess image for model input.
    
    Args:
        image_path: Path to image file
        input_hw: Tuple of (height, width)
    
    Returns:
        Preprocessed image array with shape [1, H, W, 3]
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_hw[1], input_hw[0]))  # PIL expects (W, H)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    return arr


def percentile_ms(x_ms, p):
    """Calculate percentile from list of millisecond values."""
    if len(x_ms) == 0:
        return None
    return float(np.percentile(np.asarray(x_ms, dtype=np.float64), p))


def calculate_score(accuracy, model_size_mb, macs_m):
    """Calculate competition score with size and MACs penalties.
    
    Score = Accuracy - 0.3 * log10(SizeMB) - 0.001 * MegaMACs
    """
    if model_size_mb < 0.0001:
        model_size_mb = 0.0001
    size_penalty = 0.3 * math.log10(model_size_mb)
    macs_penalty = 0.001 * macs_m
    return accuracy - size_penalty - macs_penalty


def evaluate_manifest(
    model_path: str,
    manifest_path: str,
    base_dir: str,
    warmup: int,
    max_images: int | None,
    rss_sample_every: int,
    num_threads: int,
):
    """Evaluate model on manifest-based test split."""
    # Initialize interpreter with thread control
    interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]  # [1, H, W, C]
    H, W = int(input_shape[1]), int(input_shape[2])

    # Load image paths from manifest
    image_rel_paths = load_manifest(manifest_path)
    total_available = len(image_rel_paths)
    
    if max_images is not None:
        image_rel_paths = image_rel_paths[:max_images + warmup]
        print(f"\n[WARN] --max_images enabled: Evaluating only {len(image_rel_paths)} images")
        print(f"[WARN] (includes {warmup} warmup images)")
        print(f"[WARN] Full split has {total_available} images - results are NOT official!\n")
    
    print(f"[INFO] Loaded {len(image_rel_paths)} images from manifest")
    print(f"[INFO] Warmup: {warmup} images")
    print(f"[INFO] Threads: {num_threads}")

    preprocess_ms = []
    infer_ms = []
    total_ms = []

    correct = 0
    total = 0
    peak_rss_mb_psutil = 0.0

    proc = psutil.Process(os.getpid()) if psutil else None

    # Single evaluation loop: accuracy on ALL images, latency on post-warmup only
    warmup_actual = min(warmup, len(image_rel_paths))
    print(f"\n[EVALUATION] Starting evaluation (first {warmup_actual} images are warmup)...")
    
    for i, rel_path in enumerate(image_rel_paths):
        full_path = os.path.join(base_dir, rel_path)
        true_label = 1 if rel_path.startswith('person/') else 0
        
        # Time only post-warmup images
        if i >= warmup_actual:
            t0 = time.perf_counter_ns()
            x = load_and_preprocess_image(full_path, (H, W))
            t1 = time.perf_counter_ns()

            interpreter.set_tensor(input_details[0]["index"], x)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])
            t2 = time.perf_counter_ns()

            # Timings in milliseconds
            pp = (t1 - t0) / 1e6
            inf = (t2 - t1) / 1e6
            tot = (t2 - t0) / 1e6
            preprocess_ms.append(pp)
            infer_ms.append(inf)
            total_ms.append(tot)
        else:
            # Warmup images: evaluate accuracy but don't time
            x = load_and_preprocess_image(full_path, (H, W))
            interpreter.set_tensor(input_details[0]["index"], x)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])

        pred = int(np.argmax(out))
        correct += int(pred == true_label)
        total += 1

        # Peak RSS sampling (only on post-warmup for consistency)
        if proc and i >= warmup_actual and ((i - warmup_actual + 1) % rss_sample_every == 0):
            try:
                rss_mb = proc.memory_info().rss / (1024 * 1024)
                peak_rss_mb_psutil = max(peak_rss_mb_psutil, rss_mb)
            except Exception:
                pass
        
        if total % 50 == 0:
            print(f"Processed {total} images...", end='\r')

    acc = correct / total if total > 0 else 0.0

    # ru_maxrss on Linux is KB; convert to MB
    peak_rss_mb_ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    peak_rss_mb = max(peak_rss_mb_ru, peak_rss_mb_psutil)

    stats = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "latency_total_ms": {
            "mean": float(np.mean(total_ms)) if total_ms else None,
            "p50": percentile_ms(total_ms, 50),
            "p90": percentile_ms(total_ms, 90),
            "p99": percentile_ms(total_ms, 99),
            "min": float(np.min(total_ms)) if total_ms else None,
            "max": float(np.max(total_ms)) if total_ms else None,
        },
        "latency_preprocess_ms": {
            "mean": float(np.mean(preprocess_ms)) if preprocess_ms else None,
            "p50": percentile_ms(preprocess_ms, 50),
            "p90": percentile_ms(preprocess_ms, 90),
            "p99": percentile_ms(preprocess_ms, 99),
        },
        "latency_infer_ms": {
            "mean": float(np.mean(infer_ms)) if infer_ms else None,
            "p50": percentile_ms(infer_ms, 50),
            "p90": percentile_ms(infer_ms, 90),
            "p99": percentile_ms(infer_ms, 99),
        },
        "peak_rss_mb": float(peak_rss_mb),
        "threads": num_threads,
        "warmup_images": warmup,
        "rss_sample_every": rss_sample_every,
    }
    
    print(f"\n[EVALUATION] Complete: {correct}/{total} correct")
    return stats


def evaluate_directory(
    model_path: str,
    data_dir: str,
    warmup: int,
    max_images: int | None,
    rss_sample_every: int,
    num_threads: int,
):
    """Evaluate model on directory-based test set (legacy mode)."""
    interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    H, W = int(input_shape[1]), int(input_shape[2])

    preprocess_ms = []
    infer_ms = []
    total_ms = []

    correct = 0
    total = 0
    peak_rss_mb_psutil = 0.0

    proc = psutil.Process(os.getpid()) if psutil else None

    # Collect all image paths
    image_paths = []
    labels = []
    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label)

    total_available = len(image_paths)
    if max_images is not None:
        image_paths = image_paths[:max_images + warmup]
        labels = labels[:max_images + warmup]
        print(f"\n[WARN] --max_images enabled: Evaluating only {len(image_paths)} images")
        print(f"[WARN] (includes {warmup} warmup images)")
        print(f"[WARN] Full dataset has {total_available} images - results are NOT official!\n")

    print(f"[INFO] Found {len(image_paths)} images")
    print(f"[INFO] Warmup: {warmup} images")
    print(f"[INFO] Threads: {num_threads}")

    # Single evaluation loop: accuracy on ALL images, latency on post-warmup only
    warmup_actual = min(warmup, len(image_paths))
    print(f"\n[EVALUATION] Starting evaluation (first {warmup_actual} images are warmup)...")
    
    for i, (path, y) in enumerate(zip(image_paths, labels)):
        # Time only post-warmup images
        if i >= warmup_actual:
            t0 = time.perf_counter_ns()
            x = load_and_preprocess_image(path, (H, W))
            t1 = time.perf_counter_ns()

            interpreter.set_tensor(input_details[0]["index"], x)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])
            t2 = time.perf_counter_ns()

            pp = (t1 - t0) / 1e6
            inf = (t2 - t1) / 1e6
            tot = (t2 - t0) / 1e6
            preprocess_ms.append(pp)
            infer_ms.append(inf)
            total_ms.append(tot)
        else:
            # Warmup images: evaluate accuracy but don't time
            x = load_and_preprocess_image(path, (H, W))
            interpreter.set_tensor(input_details[0]["index"], x)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])

        pred = int(np.argmax(out))
        correct += int(pred == y)
        total += 1

        if proc and i >= warmup_actual and ((i - warmup_actual + 1) % rss_sample_every == 0):
            try:
                rss_mb = proc.memory_info().rss / (1024 * 1024)
                peak_rss_mb_psutil = max(peak_rss_mb_psutil, rss_mb)
            except Exception:
                pass
        
        if total % 50 == 0:
            print(f"Processed {total} images...", end='\r')

    acc = correct / total if total > 0 else 0.0

    peak_rss_mb_ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    peak_rss_mb = max(peak_rss_mb_ru, peak_rss_mb_psutil)

    stats = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "latency_total_ms": {
            "mean": float(np.mean(total_ms)) if total_ms else None,
            "p50": percentile_ms(total_ms, 50),
            "p90": percentile_ms(total_ms, 90),
            "p99": percentile_ms(total_ms, 99),
            "min": float(np.min(total_ms)) if total_ms else None,
            "max": float(np.max(total_ms)) if total_ms else None,
        },
        "latency_preprocess_ms": {
            "mean": float(np.mean(preprocess_ms)) if preprocess_ms else None,
            "p50": percentile_ms(preprocess_ms, 50),
            "p90": percentile_ms(preprocess_ms, 90),
            "p99": percentile_ms(preprocess_ms, 99),
        },
        "latency_infer_ms": {
            "mean": float(np.mean(infer_ms)) if infer_ms else None,
            "p50": percentile_ms(infer_ms, 50),
            "p90": percentile_ms(infer_ms, 90),
            "p99": percentile_ms(infer_ms, 99),
        },
        "peak_rss_mb": float(peak_rss_mb),
        "threads": num_threads,
        "warmup_images": warmup,
        "rss_sample_every": rss_sample_every,
    }
    
    print(f"\n[EVALUATION] Complete: {correct}/{total} correct")
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Raspberry Pi Scoreboard for VWW Models",
        epilog="Supports both manifest-based splits and directory-based evaluation"
    )
    ap.add_argument("--model", required=True, help="Path to .tflite model")
    ap.add_argument("--split", type=str, default=None,
                    choices=["test_public", "val"],
                    help="Manifest-based split to evaluate (requires --splits_dir)")
    ap.add_argument("--data", type=str, default=None,
                    help="Path to test folder with person/ and non_person/ (legacy mode)")
    ap.add_argument("--base_dir", type=str, default=BASE_DIR,
                    help=f"Base directory for images (default: {BASE_DIR})")
    ap.add_argument("--splits_dir", type=str, default=SPLITS_DIR,
                    help=f"Directory containing manifest files (default: {SPLITS_DIR})")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup images (not timed)")
    ap.add_argument("--max_images", type=int, default=0, help="Limit #images timed (0 = all)")
    ap.add_argument("--threads", type=int, default=1, help="TFLite interpreter threads")
    ap.add_argument("--rss_sample_every", type=int, default=10, help="Sample RSS every N images")
    ap.add_argument("--compute_score", action="store_true", help="Compute competition score")
    ap.add_argument("--macs", type=float, default=None, help="Manual MACs override (MegaMACs)")
    ap.add_argument("--out", default="", help="Optional output JSON path")
    ap.add_argument("--official", action="store_true", 
                    help="OFFICIAL MODE: Use test_hidden for final competition scoring (instructor only)")
    args = ap.parse_args()

    model_path = args.model
    max_images = None if args.max_images == 0 else args.max_images

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    # Handle official mode (test_hidden)
    if args.official:
        args.split = "test_hidden"
        # Force consistent threading for official evaluation
        if args.threads != 1:
            print(f"[WARN] Official mode requires --threads=1 for fairness. Overriding --threads={args.threads} to 1.")
        args.threads = 1  # Always enforce threads=1 in official mode
        
        print("\n" + "="*60)
        print("🔒 OFFICIAL EVALUATION MODE - TEST_HIDDEN")
        print("="*60)
        print("This mode is RESTRICTED to final competition scoring.")
        print("test_hidden is WRITE-PROTECTED and must NEVER be used for:")
        print("  - Development")
        print("  - Model selection")
        print("  - Hyperparameter tuning")
        print("  - Any form of training or validation")
        print("Settings enforced for fairness:")
        print(f"  - Threads: {args.threads} (locked)")
        print(f"  - Warmup: {args.warmup} images")
        print("="*60 + "\n")

    # Prevent manual access to test_hidden without --official flag
    if args.split == "test_hidden" and not args.official:
        print("\n" + "="*60)
        print("[ERROR] Access to test_hidden requires --official flag")
        print("="*60)
        print("test_hidden is restricted to official competition scoring.")
        print()
        print("To access test_hidden, use:")
        print("  python src/scoreboard.py --model <model> --official --compute_score")
        print()
        print("For development, use:")
        print("  --split val        (for development and model selection)")
        print("  --split test_public (for final evaluation before submission)")
        print("="*60)
        return 1  # Exit with error code

    # Determine evaluation mode
    use_manifest = args.split is not None
    
    if use_manifest:
        manifest_path = os.path.join(args.splits_dir, f"{args.split}.txt")
        if not os.path.exists(manifest_path):
            print(f"[ERROR] Manifest not found: {manifest_path}")
            print("Run create_main_datasplit.py first to generate splits.")
            return 1
        
        # Validate test set usage
        if args.split == "test_public":
            print("\n" + "="*60)
            print("⚠️  WARNING: EVALUATING ON TEST_PUBLIC")
            print("="*60)
            print("test_public should only be used for:")
            print("  - Final model evaluation before submission")
            print("  - Comparing your best models")
            print("\nDo NOT use test_public for:")
            print("  - Training")
            print("  - Validation during development")
            print("  - Hyperparameter tuning")
            print("  - Model selection during development")
            print("\nUse 'val' split for development and model selection.")
            print("="*60 + "\n")
    else:
        if not args.data:
            print("[ERROR] Must specify either --split or --data")
            return 1
        if not os.path.exists(args.data):
            print(f"[ERROR] Data directory not found: {args.data}")
            return 1

    # Capture device info
    device = read_device_info()
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Print device info
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    for key, value in device.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")

    # Run evaluation
    if use_manifest:
        stats = evaluate_manifest(
            model_path=model_path,
            manifest_path=manifest_path,
            base_dir=args.base_dir,
            warmup=args.warmup,
            max_images=max_images,
            rss_sample_every=args.rss_sample_every,
            num_threads=args.threads,
        )
        eval_mode = f"manifest:{args.split}"
    else:
        stats = evaluate_directory(
            model_path=model_path,
            data_dir=args.data,
            warmup=args.warmup,
            max_images=max_images,
            rss_sample_every=args.rss_sample_every,
            num_threads=args.threads,
        )
        eval_mode = f"directory:{args.data}"

    # Build report
    report = {
        "model_file": os.path.basename(model_path),
        "model_size_mb": float(size_mb),
        "eval_mode": eval_mode,
        "device": device,
        "timestamp": time.ctime(),
    }
    report.update(stats)

    # Compute score if requested
    score = None
    macs_source = None
    if args.compute_score:
        # Try to load MACs from metadata JSON
        json_path = model_path.replace(".tflite", ".json")
        macs_m = None
        
        # Priority 1: Command-line flag
        if args.macs is not None:
            macs_m = args.macs
            macs_source = "command-line flag"
            print(f"[INFO] Using MACs from --macs flag: {macs_m} M")
        # Priority 2: JSON metadata file
        elif os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                macs_m = metadata.get("macs_m")
                if macs_m is not None:
                    macs_source = f"metadata file ({os.path.basename(json_path)})"
                    print(f"[INFO] Loaded MACs from {json_path}: {macs_m} M")
                else:
                    print(f"[WARN] Found {json_path} but 'macs_m' field is missing")
            except Exception as e:
                print(f"[WARN] Failed to read {json_path}: {e}")
        
        # Fail if MACs not found
        if macs_m is None:
            print("\n" + "="*60)
            print("[ERROR] Cannot compute score: MACs not provided")
            print("="*60)
            print("Score computation requires MACs. Provide them via:")
            print(f"  1. --macs <value> flag")
            print(f"  2. Metadata JSON file: {json_path}")
            print()
            print("To generate metadata JSON, run on cluster:")
            print(f"  python src/evaluate_vww.py --model {os.path.basename(model_path)} \\")
            print(f"      --split test_public --compute_score --export_json")
            print("="*60)
            return 1  # Exit with error code
        
        score = calculate_score(stats["accuracy"], size_mb, macs_m)
        report["macs_m"] = macs_m
        report["macs_source"] = macs_source
        report["score"] = score

    # Save report
    out_path = args.out.strip()
    if not out_path:
        out_path = str(Path(model_path).with_suffix(".pi_scoreboard.json"))

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Pretty print results
    print("\n" + "="*60)
    print("RASPBERRY PI SCOREBOARD REPORT")
    print("="*60)
    print(f"Model:        {report['model_file']}")
    print(f"Size:         {report['model_size_mb']:.4f} MB")
    print(f"Eval Mode:    {eval_mode}")
    print("-" * 60)
    print(f"Accuracy:     {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    print(f"Images:       {report['correct']}/{report['total']}")
    print("-" * 60)
    
    # Highlight official latency metric
    p90_latency = report['latency_total_ms']['p90']
    print(f"LATENCY (OFFICIAL): {p90_latency:.2f} ms (p90)")
    print()
    print("Detailed Latency Statistics (Total = Preprocess + Inference):")
    print(f"  Total:      p50={report['latency_total_ms']['p50']:.2f}ms  "
          f"p90={report['latency_total_ms']['p90']:.2f}ms  "
          f"p99={report['latency_total_ms']['p99']:.2f}ms")
    print(f"  Preprocess: p50={report['latency_preprocess_ms']['p50']:.2f}ms  "
          f"p90={report['latency_preprocess_ms']['p90']:.2f}ms  "
          f"p99={report['latency_preprocess_ms']['p99']:.2f}ms")
    print(f"  Inference:  p50={report['latency_infer_ms']['p50']:.2f}ms  "
          f"p90={report['latency_infer_ms']['p90']:.2f}ms  "
          f"p99={report['latency_infer_ms']['p99']:.2f}ms")
    print(f"Peak RSS:     {report['peak_rss_mb']:.2f} MB")
    print(f"Threads:      {report['threads']}")
    
    if score is not None:
        print("-" * 60)
        print(f"MACs:         {report['macs_m']:.4f} M (from {macs_source})")
        print(f"SCORE:        {score:.4f}")
    
    print("="*60)
    
    # Warnings
    if max_images is not None:
        print("\n⚠️  WARNING: --max_images was used - results are NOT official!")
        print(f"   Only {report['total']} images evaluated (partial dataset)")
    
    # Pass/Fail check
    if report['accuracy'] < 0.80:
        print("[FAIL] Accuracy is below 80% threshold.")
    else:
        print("[PASS] Meets accuracy requirement.")
    
    print(f"\nReport saved to: {out_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
