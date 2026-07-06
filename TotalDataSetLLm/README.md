# Two-Stage Traffic Sign Recognition Using CNN Ensemble and LLM-Based Semantic Verification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-GTSRB%2043%20Classes-green)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-99.52%25%20TTA-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**Authors:** Sakilam Bhargav (23211A67A7) · Moguloju Abhiram (23211A6774)  
**Supervisor:** Ramdas Sir | B.Tech 3rd Year — CSE (Data Science)  
**Dataset:** GTSRB — German Traffic Sign Recognition Benchmark (43 Classes)   
**Best Accuracy:** 99.52% (TTA Ensemble) · 99.37% (Standard Soft Voting)

---

## Key Notebooks

| Notebook | Platform | Purpose |
|----------|----------|---------|
| [`colab_llm.ipynb`](colab_llm.ipynb) | Google Colab | Full two-stage pipeline — training, batch evaluation, LLM verification, all visualizations. Mounts Google Drive automatically. |
| [`only the final model and the ouptut.ipynb`](only%20the%20final%20model%20and%20the%20ouptut.ipynb) | Local / Jupyter | Standalone inference notebook — loads the saved ensemble and runs the complete two-stage predictor on new images with LLM verification output. |

> **Start here →** Open `colab_llm.ipynb` in Google Colab for the full pipeline, or open `only the final model and the ouptut.ipynb` locally for a clean single-inference demo.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/gtsrb-cnn-llm-verification.git
cd gtsrb-cnn-llm-verification

# 2. Install dependencies
pip install torch torchvision openai requests numpy pandas matplotlib seaborn scikit-learn tqdm pillow

# 3. Set your OpenRouter API key (optional — uses offline mock if not set)
set OPENROUTER_API_KEY=your_key_here       # Windows
# export OPENROUTER_API_KEY=your_key_here  # Linux/macOS/Colab

# 4. Open the inference notebook
jupyter notebook "only the final model and the ouptut.ipynb"
```

**Google Colab:** Upload `colab_llm.ipynb` directly to Colab, copy model `.pth` files to your Google Drive under `GTSRB_TwoStage/`, then run all cells.

---

## Overview

This repository implements a **two-stage traffic sign recognition system** trained and evaluated on the complete **GTSRB dataset (43 classes)**. The system combines an **IEEE Ensemble CNN** as the primary classifier with an **LLM-based semantic verifier** that validates uncertain predictions using traffic sign visual hierarchy knowledge — operating on text only (no images sent to the LLM).

---

## Architecture — Hybrid CNN-LLM Traffic Sign Recognition System

```
                        Load Dataset
                      GTSRB (43 Classes)
                             │
                             ▼
                    Preprocess Images
                  Resize + Normalize + Augment
                             │
                             ▼
                    Train 3 CNN Models
               ConvNeXt-Tiny │ ResNet-18 │ EfficientNet-B0
                             │
                             ▼
                  Test Image — CNN Ensemble
                     Average Predictions
                             │
                             ▼
                  Get Top-3 & Compute Entropy
                             │
                    ┌────────┴────────┐
                  Yes                No
            High Confidence?
            Top-1 ≥ 85%
            Entropy ≤ 0.50
                    │                 │
                    ▼                 ▼
         ┌──────────────┐    Send Summary to LLM
         │ Accept Top-1 │           │
         │  CNN DIRECT  │           ▼
         │   LOW RISK   │   LLM Semantic Validation
         └──────┬───────┘           │
                │        ┌──────────┼──────────┐
               END       ▼          ▼          ▼
                       AGREE    DISAGREE   UNCERTAIN
                     Accept    Reject,    Accept with
                      Top-1    Review      Warning
                   CNN_LLM_  CNN_LLM_    CNN_LLM_
                    AGREED   DISAGREED   UNCERTAIN
```

---

## Algorithm

> The full 13-step algorithm is also available in [`papers/algorithm.txt`](../papers/algorithm.txt).

**Step 1.** Load the GTSRB dataset (43 traffic sign classes) with predefined train/test splits.

**Step 2.** Preprocess all images: resize to **32×32 pixels**, apply Z-score normalisation using dataset mean and standard deviation. During training apply class-distribution rebalancing and data augmentation (rotation, brightness, contrast adjustments).

**Step 3.** Develop three modified CNN architectures — **ConvNeXt-Tiny**, **ResNet-18**, and **EfficientNet-B0** — each adapted to preserve spatial detail for small 32×32 inputs by modifying stride parameters and limiting excessive early downsampling.

**Step 4.** Train all three models independently on the training split. Each model outputs a probability distribution across 43 classes.

**Step 5.** Pass the preprocessed test image through all three models. Compute the **ensemble output** by averaging the class-wise probability distributions (soft voting) to form a single unified distribution.

**Step 6.** From the ensemble distribution identify the **Top-1**, **Top-2**, and **Top-3** predicted classes with their confidence scores.

**Step 7.** Compute the **Shannon Entropy** of the full 43-class distribution to measure prediction dispersion. Compute the **confidence margin** as Top-1 score minus Top-2 score.

**Step 8.** Evaluate the confidence gate:
- If **Top-1 confidence ≥ 0.85** AND **entropy ≤ 0.50** → go to Step 9.
- Otherwise → go to Step 10.

**Step 9.** *(CNN-DIRECT path)* Accept Top-1 as the final prediction. Record decision path as **CNN_DIRECT**, assign **LOW** risk. → End.

**Step 10.** Classify the prediction as uncertain. Initiate LLM semantic validation (Stage 3).

**Step 11.** Construct a structured textual summary containing: Top-3 classes + confidence scores, entropy, margin, and semantic group definitions (prohibitory, warning, mandatory, speed limit, priority) plus reasoning rules for within-group vs. cross-group conflict detection.

**Step 12.** Send **only the text description** to the LLM (no image data transmitted). The LLM evaluates logical consistency, detects structural conflicts between predicted classes, identifies fine-grained confusion, and assesses overconfidence or out-of-distribution behaviour.

**Step 13.** Receive the LLM verdict and record final decision:

| LLM Decision | Action | Decision Path |
|---|---|---|
| **AGREE** | Accept Top-1 prediction | `CNN_LLM_AGREED` |
| **DISAGREE** | Reject prediction → flag as UNVERIFIED for human review | `CNN_LLM_DISAGREED` |
| **UNCERTAIN** | Accept Top-1 with warning + risk level | `CNN_LLM_UNCERTAIN` |

---

## Files

| File | Description |
|------|-------------|
| `total_data_llm.ipynb` | Main notebook — full pipeline: training evaluation, batch testing, visualization |
| `model_architectures.py` | CNN model class definitions (ConvNeXtTiny, ResNet18, EfficientNet-B0) |
| `llm_reviewer.py` | LLM reviewer module — structured prompt creation, OpenRouter API call, mock fallback |
| `ensemble_config.pth` | Bundled ensemble checkpoint (all 3 model weights + accuracy metadata) |
| `ieee_model_convnext_tiny.pth` | Individual ConvNeXt-Tiny checkpoint |
| `ieee_model_efficientnet_b0.pth` | Individual EfficientNet-B0 checkpoint |
| `ieee_model_resnet18.pth` | Individual ResNet18 checkpoint |
| `only the final model and the ouptut.ipynb` | Standalone inference notebook for the final model |
| `colab_llm.ipynb` | Google Colab-compatible version of the pipeline |
| `final_results.csv` | Per-image prediction results (standard) |
| `final_results_with_tta.csv` | Results with Test-Time Augmentation (TTA) |
| `model_comparison.csv` | Accuracy comparison across individual models |
| `comprehensive_model_comparison.csv` | Extended metrics across all models |
| `per_class_performance.csv` | Per-class accuracy for the ensemble |
| `per_class_all_models.csv` | Per-class accuracy breakdown for all models |

### Result Visualizations

| File | Description |
|------|-------------|
| `training_curves.png` | Loss/accuracy curves during training |
| `confusion_matrices.png` | Confusion matrix per model |
| `all_confusion_matrices.png` | All models side-by-side |
| `ensemble_confusion_matrix.png` | Ensemble-specific confusion matrix |
| `tta_vs_standard_confusion_matrix.png` | Comparison: TTA vs standard inference |
| `roc_curves_*.png` | ROC curves per model |
| `roc_comparison_all_models.png` | Overlaid ROC comparison |
| `auc_heatmap_all_models.png` | Per-class AUC heatmap across models |
| `pr_curves_*.png` | Precision-Recall curves per model |
| `pr_comparison_all_models.png` | Overlaid PR comparison |
| `radar_chart_comparison.png` | Radar chart of multi-metric model comparison |

---

## Model Architectures (`model_architectures.py`)

All three models are adapted for **32×32 GTSRB input** (stride-1 first convolution, reduced downsampling):

### 1. ConvNeXtTinyTraffic
- Base: `torchvision.models.convnext_tiny` (ImageNet pretrained)
- Modification: Stem conv changed from stride-4 to stride-1 `Conv2d(3, 96, 3, stride=1, padding=1)`
- Classifier head: `Flatten → LayerNorm → Linear(→512) → GELU → Dropout(0.3) → Linear(→43)`

### 2. EnhancedResNet18
- Base: `torchvision.models.resnet18` (ImageNet pretrained)
- Modifications: `conv1` stride-1, `maxpool` replaced with `nn.Identity()`
- Classifier head: `AdaptiveAvgPool2d → Flatten → Linear(→512) → ReLU → Dropout(0.4) → Linear(→256) → ReLU → Dropout(0.25) → Linear(→43)`

### 3. EfficientNetB0Traffic
- Base: `torchvision.models.efficientnet_b0` (ImageNet pretrained)
- Modification: Stem conv changed to stride-1 `Conv2d(3, 32, 3, stride=1, padding=1)`
- Classifier head: `Dropout(0.3) → Linear(→512) → SiLU → Dropout(0.2) → Linear(→43)`

**Ensemble:** Soft-voting (average of softmax probabilities, equal weights = 1/3 each).

---

## LLM Reviewer (`llm_reviewer.py`)

### Purpose
Provides a second opinion on uncertain CNN predictions by reasoning about **semantic consistency** of traffic sign categories — without ever processing images.

### API
- **Provider:** [OpenRouter](https://openrouter.ai) (OpenAI-compatible)
- **Default model:** `meta-llama/llama-3.1-8b-instruct`
- **Fallbacks:** `deepseek/deepseek-chat`, `google/gemma-2-9b-it:free`
- **Temperature:** `0.1` (deterministic semantic reasoning)
- **Max tokens:** `500`
- **API key:** Set env var `OPENROUTER_API_KEY`

### GTSRB Semantic Groups (used for reasoning)

| Group | Classes | Visual Shape |
|-------|---------|--------------|
| Speed Limits | 0–8, 32 | Circular, red border, white background |
| Prohibitory | 9, 10, 15, 16, 17, 41, 42 | Circular, red border |
| Priority/Stop/Yield | 11, 12, 13, 14 | Distinctive shapes (diamond, octagon, inverted triangle) |
| Warning/Danger | 18–31 | Triangular, red border |
| Mandatory | 33–40 | Circular, blue background |

### LLM Decision Types

| Decision | Confusion Type | Meaning | Risk |
|----------|---------------|---------|------|
| AGREE | `FINE_GRAINED` | Both top predictions in same semantic group | LOW |
| AGREE | `SEMANTIC_VALIDATED` | Low confidence but within-group confusion is plausible | MEDIUM |
| AGREE | `DOMINANT_PREDICTION` | High confidence + large margin — top-2 is noise | LOW |
| AGREE | `CLEAR_WINNER` | Confidence > 0.70 and margin > 0.30 | LOW |
| AGREE | `SEMANTIC_VALIDATED` | High entropy but same-group — uncertainty is appropriate | MEDIUM |
| DISAGREE | `STRUCTURAL_CONFLICT` | Different shape categories, tight margin (< 0.15) | HIGH |
| DISAGREE | `OUT_OF_DISTRIBUTION` | Near-equal probs, unrelated groups, confidence < 0.55 | CRITICAL |
| DISAGREE | `OVERCONFIDENCE` | Confidence > 0.97 but cross-group top-2 present | HIGH |
| UNCERTAIN | `AMBIGUOUS` | None of the 8 patterns match — insufficient evidence | MEDIUM |

> **Note:** Live API prompt uses 5 condensed patterns. Offline `llm_reviewer_mock()` implements all 8 patterns above with deterministic rule-based logic — no API call required.

### Offline Mock Fallback
When no API key is available, `llm_reviewer_mock()` runs the same 8-pattern semantic reasoning logic locally using deterministic rule-based matching — identical JSON output schema, no API call required. This ensures the system works fully offline, which is critical for vehicle deployment.

---

## Pipeline (Notebook: `total_data_llm.ipynb`)

| Cell | Purpose |
|------|---------|
| Cell 1 | System check, imports, device setup |
| Cell 2 | Configuration — paths, LLM API config, class names |
| Cell 3 | Model architecture definitions (embedded) |
| Cell 4 | Load ensemble models from `ensemble_config.pth` or individual `.pth` files |
| Cell 5 | LLM Reviewer setup (OpenAI-compatible client, auto-fallback) |
| Cell 6 | Ensemble prediction function — soft voting + entropy + margin |
| Cell 7 | Confidence gate function |
| Cell 8 | Complete `two_stage_predict()` pipeline function |
| Cell 9 | Quick single-image test |
| Cell 13A | LLM verification test on a random image |
| Batch cells | Full test-set evaluation with optional ground-truth CSV comparison |
| Visualization cells | Confusion matrices, ROC/PR curves, radar chart, AUC heatmap |

---

## Configuration

### LLM API (edit in Cell 2 of notebook)
```python
LLM_API_CONFIG = {
    'provider': 'openrouter',
    'base_url': 'https://openrouter.ai/api/v1',
    'api_key_env': 'OPENROUTER_API_KEY',
    'primary_model': 'meta-llama/llama-3.1-8b-instruct',
    'fallback_models': ['deepseek/deepseek-chat', 'google/gemma-2-9b-it:free'],
    'temperature': 0.1,   # Low temperature for deterministic semantic reasoning
    'max_tokens': 500,
}
```

### Confidence Gate Thresholds
```python
confidence_threshold = 0.85   # CNN confidence must exceed this
entropy_threshold    = 0.50   # Shannon entropy must be below this
```

### Image Preprocessing
```python
transforms.Resize((32, 32))
transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
                     std=[0.2672, 0.2564, 0.2629])  # GTSRB-specific
```

---

## Dataset

- **Dataset:** [GTSRB — German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html)
- **Classes:** 43 (speed limits, prohibitory, warning, priority, mandatory)
- **Input size:** 32×32 RGB
- **Split:** `model_data/train/` and `model_data/test/` (class subfolders `0/` to `42/`)

---

## Usage

### Single Image Prediction
```python
result = two_stage_predict('path/to/image.png', use_llm=True, verbose=True)
print(result['final_prediction']['class_name'])
print(result['decision_path'])  # CNN_DIRECT | CNN_LLM_AGREED | CNN_LLM_DISAGREED
```

### Batch Evaluation
```python
# Without ground truth
test_all_images()

# With ground truth CSV (columns: image_name, class_id)
test_all_images(ground_truth_csv='test_labels.csv')

# Limit sample size
test_all_images(max_images=200)
```

### Direct LLM Mock Review
```python
from llm_reviewer import llm_reviewer_mock

cnn_output = {
    "predictions": [
        {"class_id": 2, "class_name": "Speed limit (50km/h)", "confidence": 0.85},
        {"class_id": 1, "class_name": "Speed limit (30km/h)", "confidence": 0.10},
        {"class_id": 3, "class_name": "Speed limit (60km/h)", "confidence": 0.03},
    ],
    "entropy": 0.45,
    "margin": 0.75,
}
response = llm_reviewer_mock(cnn_output)
# → {"decision": "AGREE", "confusion_type": "DOMINANT_PREDICTION", ...}
```

---

## Environment

- Python 3.8+
- PyTorch + torchvision
- `openai` (for OpenAI-compatible API client)
- `requests`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`, `PIL`

**Colab:** Use `colab_llm.ipynb` — automatically mounts Google Drive and locates models from `/content/drive/MyDrive/GTSRB_TwoStage`.

---

## Results Summary

### Individual Model Performance (12,630 test images)

| Model | Accuracy (%) | Macro F1 (%) | ROC-AUC | Params | Train Time |
|-------|-------------|-------------|---------|--------|------------|
| ConvNeXt-Tiny | **99.41** | 99.11 | **1.0000** | 28.23M | 63.9 min |
| Enhanced ResNet-18 | 99.11 | 98.36 | 0.9998 | 11.57M | **37.2 min** |
| EfficientNet-B0 | 98.16 | 97.18 | 0.9999 | **4.69M** | 75.8 min |

### Ensemble Performance

| Configuration | Accuracy (%) | Top-3 Acc (%) | Macro F1 (%) | Cohen Kappa | Log Loss |
|---------------|-------------|--------------|-------------|------------|----------|
| Soft Voting | 99.37 | **99.90** | 98.92 | 0.9934 | 0.2532 |
| TTA Ensemble | **99.52** | — | **99.19** | — | — |

> TTA augmentation strategy follows Deepika et al. (2023), applying the same geometric and photometric transforms at inference and averaging predictions across augmented copies.

### Most Challenging Classes

| Class | Sign | ConvNeXt F1 | ResNet-18 F1 | EfficientNet F1 | Root Cause |
|-------|------|-------------|--------------|-----------------|------------|
| 20 | Dangerous curve right | 95.74% | 89.55% | 87.38% | Within-group confusion with Class 21 |
| 21 | Double curve | 93.75% | 91.84% | 88.67% | Within-group confusion with Class 20 |
| **42** | **End no passing (>3.5t)** | 98.90% | 94.80% | **86.01%** | Fine stripe pattern lost at 32×32 |
| 6 | End of 80km/h limit | 98.65% | 95.47% | 89.30% | Diagonal strikethrough detail lost |
| 30 | Beware of ice/snow | 96.05% | 90.85% | 90.66% | Warning-group pictogram similarity |

Class 20/21 confusion is `FINE_GRAINED` (LLM Risk: LOW) — both are triangular warning signs. Class 42 and Class 6 failures on EfficientNet are structural — fine detail is lost during downsampling to 32×32, an inherent limitation of the smallest model.

---

## Key Design Decisions

1. **Text-only LLM input** — The LLM never receives image data; it only reasons about class names, confidence scores, entropy, and margin. This keeps inference fast and cost-efficient.
2. **Negligible alternative filtering** — If Top-2 confidence < 5% or margin > 50%, the LLM is instructed to ignore cross-group alternatives as noise (not overconfidence).
3. **Mock fallback** — `llm_reviewer_mock()` mirrors the real LLM reasoning rules locally, enabling offline development and testing.
4. **Stride-1 stem modification** — All three pretrained ImageNet models are adapted for small 32×32 images by replacing the default stride-4/stride-2 first convolution with stride-1 to prevent excessive spatial downsampling.
5. **Parameter counts** — Computed using `sum(p.numel() for p in model.parameters())` in PyTorch over all layers including modified stems and custom heads (ConvNeXt: 28.23M, ResNet-18: 11.57M, EfficientNet-B0: 4.69M).
