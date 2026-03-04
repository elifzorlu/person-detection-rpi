# Person Detection on Raspberry Pi — Compressed MobileNetV1

Compressed and deployed a person detection model on Raspberry Pi using the Visual Wake Words (VWW) dataset. Starting from a MobileNetV1 baseline (~86% accuracy, ~1.5 MB), applied a combination of architecture scaling, knowledge distillation, and INT8 quantization to produce a model that is **8× smaller** and **6× fewer MACs** while retaining 82.9% accuracy.

## Results

| Metric | Baseline | This Model |
|---|---|---|
| Accuracy | ~86% | **82.91%** |
| Model Size | ~1.5 MB | **0.193 MB** |
| MACs | ~27 M | **4.456 M** |
| Latency (p90) | — | **2.89 ms/image** |
| Competition Score | — | **1.0388** |

Score formula: `Accuracy − 0.3×log₁₀(Size_MB) − 0.001×MACs_M`

## Techniques

### 1. Width Scaling (Architecture Reduction)
Reduced the MobileNetV1 width multiplier from α≈0.25 to α≈0.19 (`num_filters` 8 → 6). This cuts MACs quadratically (~0.57×) while preserving the depthwise-separable convolution structure. All ReLU activations were replaced with **ReLU6** to clamp the activation range to [0, 6], reducing quantization error during INT8 conversion.

### 2. Knowledge Distillation (Warm-Start)
Trained the compressed student model against a frozen teacher using a combined loss:

```
Loss = α · CE(labels, student) + (1−α) · T² · KL(soft_teacher ∥ soft_student)
```

- **Temperature T = 2.0**: binary classification produces peaked distributions; high T over-smooths the signal
- **α = 0.7**: weights ground truth more than teacher to reduce gradient noise from teacher errors
- **Warm-start**: initialized student from a pre-trained checkpoint rather than random weights, preventing the KLD/CE conflict that occurs with random initialization

### 3. INT8 Post-Training Quantization
Converted to TFLite with full INT8 quantization (`Optimize.DEFAULT + TFLITE_BUILTINS_INT8`). Used 1000 randomly sampled calibration images (instead of the default 200) to better cover the input distribution and reduce per-channel quantization error.

## Tradeoffs

- **Size vs. accuracy**: Going from 6 → 8 filters would increase accuracy potential to ~85%+ but raises model size to ~0.25 MB. At that size, you need 87%+ accuracy just to match the current score — not worth it.
- **Quantization drop**: INT8 costs ~0.4 pp (83.3% float → 82.9% INT8), acceptable for 4× size savings.
- **Pruning discarded**: TFLite INT8 does not exploit weight sparsity for file size reduction, so magnitude pruning reduced accuracy with no size benefit.

## Project Structure

```
├── src/
│   ├── vww_model.py          # MobileNetV1 architecture (width scaling, relu6)
│   ├── train_vww.py          # Baseline training script
│   ├── train_distill.py      # Knowledge distillation training
│   ├── convert_to_tflite.py  # INT8 TFLite conversion
│   └── evaluate_vww.py       # Evaluation + scoring
├── submission/
│   ├── model.tflite          # Final compressed model (not tracked, see below)
│   ├── model.json            # Evaluation metrics
│   └── README_short.md       # Summary report
└── splits/
    ├── train.txt
    ├── val.txt
    └── test_public.txt
```

> **Note:** The dataset (`vw_coco2014_96/`) and model weights (`.h5`, `.tflite`) are excluded from this repo via `.gitignore` due to size. The `submission/model.tflite` file is available separately.

## Setup

```bash
conda env create -f env.yml
conda activate vww_env
export LD_LIBRARY_PATH=$HOME/.conda/envs/vww_env/lib:/usr/lib/x86_64-linux-gnu
```

## Reproducing

```bash
# Train with distillation
python src/train_distill.py      # → trained_models/vww_96_relu6_distill.h5

# Convert to INT8 TFLite
python src/convert_to_tflite.py  # → models/vww_96_relu6_distill_int8.tflite

# Evaluate
python src/evaluate_vww.py --model models/vww_96_relu6_distill_int8.tflite \
  --split test_public --compute_score --export_json
```

## Dataset

Visual Wake Words — binary person/non-person classification on 96×96 images, derived from MS COCO 2014. Download from [SiLabs](https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz).
