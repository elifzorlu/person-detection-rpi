# ECE 341X — Final Project
## Visual Wake Words on Raspberry Pi

This project implements a Person Detection model using Visual Wake Words (VWW) dataset with MobileNetV1 architecture. The goal is to compress and deploy this model for efficient execution on a Raspberry Pi while maintaining strong accuracy (≥80%).

## Project Overview

You are given:
- **Dataset:** Visual Wake Words (VWW), binary classification (`person` / `non_person`)
- **Base Model:** MobileNetV1 (baseline accuracy ≈ 86%)
- **Target Platform:** Raspberry Pi (CPU-only, TensorFlow Lite)

Your task is to **compress and deploy** this model while balancing:
- High accuracy (minimum 80% on hidden test set)
- Small model size
- Low computational complexity (MACs)
- Fast inference latency

A leaderboard will rank submissions based on a composite score. Latency will be graded separately.

## Quick Start: Environment Setup
The setup of the environment about how to use the campus cluster is same as Lab 1 and Lab2. The main thing to focus on are the TF and Cuda versions that are compatible with the provided code. 
To avoid version conflicts (especially with Keras 3 or GLIBCXX), use the provided conda environment:

```bash
module load gcc
module load cuda
conda env create -f env.yml python=3.11.7
conda activate vww_env
```

DO NOT RUN: 
```bash
module load miniconda
```

Students are most welcome to have different TF and Cuda version but note that many functions provided will not work out of the box and 

## Dataset Preparation

We use the vw_coco2014_96 dataset containing pre-cropped 96x96 images suitable for MobileNet architecture.

### Dataset Structure

```
vw_coco2014_96/
├── person/
└── non_person/
```

### 1. Download Dataset

```bash
wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
tar -xvf vw_coco2014_96.tar.gz
```

### 2. Deterministic Data Splits

All manifest files are in the `splits/` directory (no files are moved):

```
splits/
├── train.txt          # Training set
├── val.txt            # Validation set
└── test_public.txt    # Public test set
```

Each manifest contains one relative file path per line, e.g.:
```
person/000123.jpg
non_person/004567.jpg
```

### Data Split Rules

**You may train and tune using:**
- `train.txt` - For model training
- `val.txt` - For development and model selection
- `test_public.txt` - For final evaluation before submission

**Leaderboard scoring uses `test_hidden.txt`, which is:**
- Disjoint from all public splits
- Instructor-only (requires `--write_hidden` flag to generate)
- Not available to students
- Used for official competition ranking

**Important:** The splits are deterministic (same seed = same split). Never train on test data!

### 3. Final Directory Structure

Ensure your directory structure looks like this:

```bash
341x_project/
├── src/
│   ├── train_vww.py
│   ├── vww_model.py
│   ├── create_main_datasplit.py
│   ├── evaluate_vww.py
│   └── scoreboard.py
├── vw_coco2014_96/
│   ├── person/
│   └── non_person/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   ├── test_public.txt
│   └── test_hidden.txt
└── models/
```

## Training

### GPU Setup on Cluster

A common issue found when testing on the campus cluster is TensorFlow not recognizing or finding the GPU libraries. Not using a GPU when training could increase training time by more than 2x. These commands helped:

```bash
# 1. Load the cluster's GPU software stack FIRST
module load cuda12.2/toolkit/12.2.2
module load cudnn8.9-cuda12.2/8.9.7.29
# Point to your conda environment's library folder
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/vww_env/lib:$LD_LIBRARY_PATH
# Help XLA find CUDA
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$(dirname $(dirname $(which nvcc)))

```

Make sure that you are using a GPU either through sinteractive or a slurm job when training!

### Run Training

```bash
python src/train_vww.py
```

The script will:
- Load training data from `splits/train.txt`
- Validate on data from `splits/val.txt`
- Save the trained model to `trained_models/vww_96.h5`

You can modify the script to save models with different names or add command-line arguments for hyperparameters.

## Evaluation and Scoring

### Development Phase: Use Validation Set

During development, always use the validation set to evaluate your models:

```bash
# On cluster (supports both .h5 and .tflite)
python src/evaluate_vww.py --model trained_models/vww_96.h5 --split val

# On Raspberry Pi
python src/scoreboard.py --model models/vww_96.tflite --split val

# Quick profiling on Pi (limited images for faster iteration)
python src/scoreboard.py --model models/my_model.tflite --split val --max_images 100
```

### Pre-Submission: Evaluate on test_public

Before submitting, evaluate your best model on the public test set:

```bash
# On cluster - with full metrics and score
python src/evaluate_vww.py --model models/vww_96.tflite --split test_public --compute_score --export_json

# On Raspberry Pi - with detailed profiling
python src/scoreboard.py --model models/vww_96.tflite --split test_public --compute_score
```

The cluster evaluation will:
- Compute accuracy on test_public
- Calculate exact MACs from the TFLite model
- Measure latency percentiles (p50/p90/p99)
- Calculate competition score
- Export metadata JSON (if accuracy >= 80%)

### Official Scoring: test_hidden (Instructor Only)

The hidden test set is ONLY used for final competition scoring:

```bash
# On Raspberry Pi (requires --official flag)
python src/scoreboard.py --model models/vww_96.tflite --official --compute_score
```

**Official Mode Enforcement:**
- Requires `--official` flag to access test_hidden
- Forces `--threads=1` for fair comparison (cannot be overridden)
- Uses fixed warmup settings
- Generates official competition results

**IMPORTANT:** 
- Use `val` split during development and model selection
- Use `test_public` only for final evaluation before submission
- `test_hidden` is restricted to official competition scoring
- Never train or tune on test data!

### Scoring Formula

$$\text{Score} = \text{Accuracy} - 0.3 \times \log_{10}(\text{ModelSize}_{\text{MB}}) - 0.001 \times \text{MACs}_{\text{M}}$$

Where:
- **Accuracy** is measured on the official hidden test set
- **ModelSize** is the `.tflite` file size in MB
- **MACs** are computed using the official scoring script

**Note:** Latency is NOT included in this formula. It will be graded separately.

### Reported Metrics

All teams must report:
- **Accuracy (%)** - Top-1 accuracy on test set
- **Latency (ms/image)** - p90 latency (official metric for grading)
- **Model Size (MB)** - TFLite file size
- **Peak RSS Memory (MB)** - Maximum memory usage
- **MACs (M)** - Mega Multiply-Accumulate operations

### Latency Definition

For each image, timing includes:
- Image load and decode
- Resize to model input resolution
- Normalization
- Tensor preparation
- `interpreter.invoke()`
- Output retrieval and argmax

**Official Latency Metric: p90 (90th percentile)**

The p90 latency is used for grading as it provides a robust measure that:
- Excludes outliers (unlike max)
- Represents typical worst-case performance
- Is more stable than mean across runs

All official latency measurements will be performed on the course Raspberry Pi using a fixed software environment with 1 thread and 50 warmup images.

### Evaluation Features

Both evaluation scripts include:
- **Warmup phase**: 50 images (not counted in metrics)
- **Latency percentiles**: p50, p90, p99
- **Latency measurement**: Full pipeline timing (load + preprocess + inference + output)
- **Fixed thread count**: Default 1 thread (configurable)
- **Memory profiling**: Peak RSS tracking
- **CPU governor check**: Recommends "performance" mode

## Grading Rubric (100 Points Total)

### A. Correct Setup & Reproducibility (15 pts)

-   Model loads successfully on evaluation Raspberry Pi (5 pts)
-   Correct `.json` MAC metadata provided (5 pts)
-   Submission folder structured correctly and reproducible (5 pts)

### B. Accuracy Performance (25 pts)

-   ≥ 80% accuracy (required threshold)

-   80--82%: 15 pts

-   82--85%: 20 pts

-   85%: 25 pts

### C. Model Efficiency (20 pts)

Based on leaderboard score components: - Model size reduction relative
to baseline - MAC reduction relative to baseline Higher efficiency
yields higher points (scaled across submissions).

### D. Latency Performance (20 pts)

Measured on official Raspberry Pi: - Excellent latency improvement: 20
pts - Moderate improvement: 15 pts - Meets baseline: 10 pts - Slower
than baseline: 5 pts

### E. Technical Understanding (20 pts)

From `README_short.md`: - Clear explanation of compression strategy (10
pts) - Insightful discussion of tradeoffs and bottlenecks (10 pts)

## Common Errors

| Error | Cause | Potential Fix |
| :--- | :--- | :--- |
| **GLIBCXX_3.4.29 not found** | OS uses old C++ libraries. | Export the `LD_LIBRARY_PATH` to your conda `/lib` folder. |
| **Unrecognized keyword arguments** | Keras 3 version mismatch. | Make sure to use the version of TensorFlow in the yml file. |
| **ModuleNotFoundError: PIL** | Pillow is missing in env. | `pip install Pillow` |
| **Skipping registering GPU...** | CUDA libraries not found. | Ensure you are on a GPU node and `LD_LIBRARY_PATH` is exported. |

Please make sure that you are reading and udnerstanding the error that are coming up. Please don't take all the commands at face value. Please look at the code and understand what is happening and why you might be getting the errors. If you need help debugging please reach out to us as soon as you can. We are always available in office hours, lab hours, and are happy to schedule more time if needed. 

## Optimization Hints

A standard "dense" model will likely get a poor score. You are expected to explore different strategies. You will need to research and modify your training/conversion code to implement these.

### Allowed Methods

You may use:
- Post-training quantization (FP16, INT8)
- Quantization-aware training
- Structured or unstructured pruning
- Knowledge distillation
- Architecture modifications
- Data Augmentation (but no data movement between splits are allowed)

You may NOT:
- Train or tune using `test_hidden.txt`
- Use GPU/NPU acceleration during official evaluation
- Change the classification task

### Weight Pruning (Sparsity)

Pruning zeros out "unimportant" weights. You can use Pruning-Aware Training (PAT) to help the model maintain accuracy while becoming sparse.

Hint: Look into the tensorflow_model_optimization (tfmot) library, specifically `prune_low_magnitude`.

### Quantization

Standard models use 32-bit floats. Converting to 8-bit integers (INT8) reduces size by 4x.

Hint: Look at the TFLiteConverter options for `optimizations` and `representative_dataset`.

## Model Conversion

The training script saves a .h5 (Keras) model, which is NOT compatible with Raspberry Pi. You must convert it to TFLite format:

```python
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('trained_models/vww_96.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Apply optimizations
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen

tflite_model = converter.convert()

# Save the model
with open('models/vww_96.tflite', 'wb') as f:
    f.write(tflite_model)
```

## About train_vww.py

The training happens in `train_vww.py`. To apply techniques learned in class, you will need to update this file.

### Script Overview

- **Configurations**: The script defines IMAGE_SIZE, BATCH_SIZE, and EPOCHS.
- **Data Pipelines**: Uses manifest-based loading from `splits/train.txt` and `splits/val.txt`
- **Main Loop**: The `main()` function loads data and orchestrates training. You can add command-line arguments (using argparse) to test different hyperparameters.
- **Training Engine**: `train_epochs()` handles compilation and fitting. Experiment with different optimizers, loss functions, and metrics here.

Note: The script now uses manifest-based splits instead of random splitting, ensuring reproducible and fair evaluation.

## Submission Requirements

Submit a folder containing:

### 1) `model.tflite`
Your final compressed deployable TensorFlow Lite model.

### 2) `model.json`
Generated using the official evaluation script:
```bash
python src/evaluate_vww.py --model model.tflite --split test_public --compute_score --export_json
```
This file must contain the verified MAC count and accuracy metrics.

### 3) `README_short.md` (≤ 1 page)
Include:
- Compression techniques used
- Final local metrics:
  - Accuracy
  - Latency
  - Model size
  - Peak memory
- Notes needed to reproduce your export process

### Hard Requirements

Your submission must satisfy:
- **Top-1 Accuracy ≥ 80%** on the official hidden test set
- Must run on Raspberry Pi CPU using TensorFlow Lite

Submissions below 80% accuracy are not eligible for leaderboard ranking.

## Official Evaluation Environment

Students may use any Raspberry Pi version (Pi 2–Pi 5) for development.

All official scoring (leaderboard + latency grade) will be conducted on a single course Raspberry Pi to ensure fairness.

**Important Notes:**
- Ensure your exported `.tflite` file is final and self-contained
- Ensure `model.json` corresponds exactly to the submitted `.tflite`
- Submissions that fail to load or run on the evaluation Pi will not be ranked
