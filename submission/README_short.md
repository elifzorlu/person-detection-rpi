# ECE 341X Final Project — README_short.md

## Compression Strategy

Three techniques were combined to maximize the competition score:
`Score = Accuracy − 0.3·log₁₀(Size_MB) − 0.001·MACs_M`

### 1. Architecture Reduction (Width Scaling)
`num_filters` in the MobileNetV1 backbone was reduced from **8 → 6**,
shrinking the width multiplier from α≈0.25 to α≈0.19. This cuts MACs
quadratically (≈0.57×) and parameter count significantly while keeping
the depthwise-separable structure intact. All ReLU activations were
replaced with **ReLU6** to clamp activations to [0, 6], which tightens
the dynamic range and reduces quantization error during INT8 conversion.

### 2. Warm-Start Knowledge Distillation
A student (num_filters=6, relu6) was initialized from a previously
trained checkpoint rather than random weights. It was then trained
against a frozen teacher (`vww_96.h5`, baseline ~86% val accuracy)
using a combined loss:

```
Loss = α · CE(labels, student) + (1−α) · T² · KL(soft_teacher ∥ soft_student)
```

Key hyperparameter choices and rationale:
- **Temperature T = 2.0** (not 4.0): binary classification produces
  already-peaked teacher distributions; high T washes out the signal.
- **α = 0.7** (ground-truth weight): teacher errs on hard negatives, so
  trusting ground truth more than the teacher reduces noise in the
  distillation gradient.
- **Math fix**: teacher outputs probabilities (final softmax), so
  `log(p + ε)` converts them back to logit-space before temperature
  scaling — avoiding log(softmax(softmax(·))) double-squashing.
- **Warm-start**: initializing from a trained checkpoint (not random
  weights) avoids the conflict between KLD and CE losses in early
  training that prevents convergence.

Training ran for 20 epochs with Adam (lr=1e-4 → 1e-5); best checkpoint
was saved by val_acc.

### 3. Post-Training INT8 Quantization
The `.h5` model was converted to TFLite with full INT8 quantization
(`Optimize.DEFAULT + TFLITE_BUILTINS_INT8`). Calibration used **1000
randomly shuffled** training images (vs. the default first-200) to
better cover the input distribution and reduce quantization error.

---

## Final Metrics (test_public)

| Metric | Value |
|---|---|
| Accuracy | **82.91%** |
| Model Size | **0.193 MB** |
| MACs | **4.456 M** |
| Latency p50 / p90 / p99 | 1.96 / 2.89 / 7.83 ms |
| Peak RSS Memory | **355.6 MB** |
| Competition Score | **1.0388** |

---

## Tradeoffs & Bottlenecks

- **Architecture ceiling**: num_filters=6 caps accuracy near 83%
  regardless of training strategy. Distillation closed the gap vs.
  vanilla training (~82% → 83%) but could not exceed the teacher.
- **Score vs. accuracy tradeoff**: Increasing to 8 filters would raise
  potential accuracy to ~85%+ but at 0.25+ MB — requiring 87%+ accuracy
  just to match the current score. The tiny 6-filter model earns
  +0.214 score points from size alone.
- **Quantization accuracy drop**: INT8 conversion costs ≈0.4 pp
  (83.4% float → 83.0% INT8), acceptable given the 4× size savings.
- **Unstructured pruning was discarded**: TFLite INT8 does not exploit
  sparse weight patterns for size reduction, so magnitude pruning gave
  no benefit in file size while hurting accuracy.

---

## Reproducing the Export

```bash
# 1. Activate environment
conda activate vww_env
export LD_LIBRARY_PATH=$HOME/.conda/envs/vww_env/lib:/usr/lib/x86_64-linux-gnu

# 2. Knowledge distillation (warm-start from relu6 checkpoint)
python src/train_distill.py      # → trained_models/vww_96_relu6_distill.h5

# 3. INT8 TFLite conversion
python src/convert_to_tflite.py  # → models/vww_96_relu6_distill_int8.tflite

# 4. Evaluate and export JSON
python src/evaluate_vww.py \
  --model models/vww_96_relu6_distill_int8.tflite \
  --split test_public --compute_score --export_json
```

**Submission files:**
- `submission/model.tflite`
- `submission/model.json`
- `README_short.md`
