# Facial Emotion Detection

A deep learning project that detects facial emotions in real time using a custom CNN trained on FER2013. The model classifies faces into 7 emotions and explains its predictions visually using Grad-CAM. A Streamlit web app lets users interact with the model via webcam.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Streamlit App](#streamlit-app)
- [DevOps & CI/CD](#devops--cicd)
- [Git Workflow](#git-workflow)
- [Team Task Split](#team-task-split)
- [Getting Started](#getting-started)

---

## Project Overview

| Item | Detail |
|------|--------|
| Task | Multi-class facial emotion classification |
| Classes | angry, disgust, fear, happy, sad, surprise, neutral |
| Dataset | FER2013 (Kaggle image folder format) |
| Model | Custom CNN from scratch (PyTorch) |
| Explainability | Custom Grad-CAM implementation |
| App | Streamlit webcam app |
| Deployment | Hugging Face Spaces via Docker |
| Experiment Tracking | Weights & Biases (wandb) |

---

## Tech Stack

| Category | Library |
|----------|---------|
| Deep Learning | PyTorch + torchvision |
| Face Detection | MediaPipe |
| Image Processing | OpenCV, Pillow |
| Data Science | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Experiment Tracking | Weights & Biases (wandb) |
| Explainability | Custom Grad-CAM (~30 lines PyTorch) |
| App | Streamlit |
| Deployment | Hugging Face Spaces, Docker |
| Model Export | TorchScript / ONNX |
| CI/CD | GitHub Actions |

---

## Project Structure

```
facial-emotion-detection/
├── app/
│   ├── streamlit_app.py          # Main Streamlit app
│   └── components/
│       ├── confidence_bar.py     # Confidence scores UI
│       ├── gradcam_view.py       # Grad-CAM heatmap UI
│       └── webcam.py             # Webcam capture + preprocessing
├── configs/
│   └── config.yaml               # All project configuration
├── data/
│   └── README.md
├── docker/
│   └── Dockerfile
├── docs/
│   └── results/                  # Confusion matrix, Grad-CAM samples
├── models/
│   └── best_model.pt             # Trained model checkpoint
├── notebooks/                    # Kaggle experiment notebooks
├── scripts/
│   ├── benchmark_inference.py
│   ├── deploy_to_hf.py
│   ├── download_checkpoint.py
│   ├── export_model.py
│   └── validate_accuracy.py
├── src/
│   ├── dataset.py                # Dataset class + dataloaders
│   ├── evaluate.py               # Metrics + confusion matrix
│   ├── gradcam.py                # Grad-CAM implementation
│   ├── model.py                  # CNN architecture
│   ├── train.py                  # Training loop
│   └── utils.py
├── tests/
│   ├── test_dataset.py
│   ├── test_gradcam.py
│   └── test_model.py
├── .github/workflows/            # CI/CD pipelines
├── docker-compose.yml
├── docker-compose.dev.yml
├── docker-compose.prod.yml
└── requirements.txt
```

---

## Dataset

**FER2013** — Facial Expression Recognition 2013 dataset from Kaggle.

- 35,887 grayscale face images at 48×48 pixels
- 7 emotion classes: angry, disgust, fear, happy, sad, surprise, neutral
- Split into `train/` and `test/` folders, each containing one subfolder per emotion

```
fer2013/
  train/
    angry/     disgust/    fear/
    happy/     neutral/    sad/     surprise/
  test/
    angry/     disgust/    fear/
    happy/     neutral/    sad/     surprise/
```

The dataset is loaded from Kaggle at:
```
/kaggle/input/datasets/msambare/fer2013
```

> FER2013 is a challenging dataset. State-of-the-art models achieve ~73% accuracy. A well-tuned CNN from scratch typically reaches 55-65%.

---

## Data Preprocessing

All preprocessing is handled in `src/dataset.py` via the `FERDataset` class and `get_transforms()` function.

**Training transforms:**
1. Resize to 48×48
2. Random horizontal flip (data augmentation)
3. Random rotation ±10° (data augmentation)
4. Convert to tensor
5. Normalize with mean=0.5077, std=0.2551 (computed from FER2013)

**Validation/Test transforms:**
1. Resize to 48×48
2. Convert to tensor
3. Normalize with mean=0.5077, std=0.2551

**Class weights:** Computed automatically using `sklearn.utils.class_weight.compute_class_weight` to handle FER2013's class imbalance (happy and neutral are overrepresented). Passed to `CrossEntropyLoss` during training.

**Why grayscale?** FER2013 images are natively grayscale — emotion is conveyed by facial geometry and texture, not color. Using grayscale reduces model complexity (1 input channel instead of 3) without losing relevant information.

---

## Model Architecture

Defined in `src/model.py` as `EmotionCNN`. A 4-block CNN built from scratch using PyTorch.

```
Input (1, 48, 48)
      ↓
Block 1 — 32 filters  → learns basic edges & textures     → 24×24
      ↓
Block 2 — 64 filters  → learns facial parts               → 12×12
      ↓
Block 3 — 128 filters → learns emotion patterns           → 6×6
      ↓
Block 4 — 256 filters → learns high-level combinations    → 3×3
      ↓
Flatten → FC(2304→1024) → Dropout(0.5) → FC(1024→7)
      ↓
Output (7 class logits)
```

**Each block follows the same pattern:**
```
Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU → MaxPool2d(2,2) → Dropout2d(0.25)
```

**Design choices:**
- **BatchNorm** after every Conv — stabilizes training and speeds convergence
- **Two convolutions per block** — increases expressiveness without adding too many parameters
- **MaxPool(2,2)** — halves spatial dimensions at each block
- **Dropout 0.5** in classifier — FER2013 overfits easily, dropout is critical
- **`build_model()` factory function** — used by `train.py`, `evaluate.py`, and the Streamlit app for consistent instantiation

---

## Training

Training is handled in `src/train.py` and runs on Kaggle (GPU).

**Configuration** (from `configs/config.yaml`):

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Max epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Early stopping patience | 10 epochs |
| Loss function | CrossEntropyLoss (weighted) |

**Training loop:**
1. Load data with `get_dataloaders()`
2. Instantiate `EmotionCNN` and move to GPU
3. Apply class weights to loss function
4. Train one epoch → log train loss + accuracy to wandb
5. Evaluate on test set → log val loss + accuracy to wandb
6. Step scheduler
7. Save checkpoint if val_acc improves (`models/best_model.pt`)
8. Early stopping if no improvement for 10 epochs

**Experiment tracking** with Weights & Biases — logs training curves, hyperparameters, and live accuracy dashboards per run.

---

## Evaluation

Handled in `src/evaluate.py`.

**`evaluate(model, loader, criterion, device)`**
Called by `train.py` every epoch. Returns average loss and accuracy on the given dataloader.

**`evaluate_model(model, loader, device, config)`**
Full post-training evaluation:
- Runs inference on the entire test set
- Generates a per-class `classification_report` (precision, recall, F1) using scikit-learn
- Computes and saves a confusion matrix heatmap to `docs/results/confusion_matrix.png`
- Reports overall accuracy

**`load_and_evaluate()`**
Standalone entry point — loads `best_model.pt` and runs full evaluation without needing to retrain.

---

## Grad-CAM Explainability

Implemented from scratch in `src/gradcam.py` (~30 lines of PyTorch). No external library used.

**What is Grad-CAM?**
Gradient-weighted Class Activation Mapping — a technique that highlights which regions of an input image were most important for a model's prediction. It answers: *"Why did the model predict 'happy'?"*

**How it works:**
1. Run a forward pass through the model
2. Compute gradients of the target class score with respect to the last convolutional layer (`block4`)
3. Pool the gradients across channels to get importance weights
4. Weight the activation maps by these importance weights
5. Apply ReLU and normalize to produce a heatmap in [0, 1]
6. Resize and overlay on the original image using a jet colormap

**Why `block4`?**
It is the last convolutional layer — it has the richest semantic features while still retaining spatial information about where in the image those features were found.

**Output:** A side-by-side visualization of the original face and the heatmap overlay, saved to `docs/results/gradcam_samples/`.

---

## Streamlit App

Located in `app/streamlit_app.py`. Run locally with:

```bash
streamlit run app/streamlit_app.py
```

**Features:**
- Webcam capture via `st.camera_input()`
- Real-time preprocessing (grayscale, resize, normalize)
- Model inference with `EmotionCNN`
- Predicted emotion displayed prominently
- Confidence scores shown as sorted progress bars for all 7 emotions
- Grad-CAM heatmap overlay shown side by side with original image

**Components:**
- `app/components/webcam.py` — `preprocess_image()` and `decode_prediction()`
- `app/components/confidence_bar.py` — `render_confidence_bar()`
- `app/components/gradcam_view.py` — `render_gradcam()`

**Requirements:** `models/best_model.pt` must exist before running the app.

---

## DevOps & CI/CD

### Docker

| File | Purpose |
|------|---------|
| `docker/Dockerfile` | Base image for the app |
| `docker-compose.yml` | Base compose config |
| `docker-compose.dev.yml` | Development overrides |
| `docker-compose.prod.yml` | Production config for HF Spaces |

### GitHub Actions Workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | Every PR to develop | Lint + unit tests |
| `docker-build.yml` | Every PR to develop | Builds Docker image |
| `model_validation.yml` | PR to main | Validates model accuracy threshold |
| `deploy.yml` | Merge to main | Deploys to Hugging Face Spaces |

### Deployment Flow
```
feature/* PR → develop   → CI runs (lint + tests + docker build)
develop PR  → main       → model validation → Docker push → HF Spaces deploy
```

---

## Git Workflow

This project uses a **3-tier branching strategy**:

| Branch | Purpose | Rules |
|--------|---------|-------|
| `main` | Production | No direct pushes. Only from develop via PR. Every commit triggers HF deploy. |
| `develop` | Integration | Never broken. Receives merges from feature/* only. CI must always pass. |
| `feature/*` | Daily work | Branch off develop. One feature per branch. Deleted after merge. |
| `experiment/*` | Throwaway | Kaggle notebooks and ideas. Never merged directly. Cherry-pick what works. |

**Rules:**
- No self-merges — A reviews B's PRs, B reviews A's PRs
- Both must approve `develop → main` merge
- Commit messages follow `feat:`, `fix:`, `config:`, `test:` prefixes

---

## Team Task Split

| Person | Responsibilities |
|--------|----------------|
| A | Data pipeline, training loop, wandb, Docker infra, CI workflows, utility scripts |
| B | CNN architecture, evaluation, Grad-CAM, Streamlit app, CD workflows, deploy scripts |

Both review each other's PRs. Both approve `develop → main`.

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/AlaeRhazouani/facial-emotion-detection.git
cd facial-emotion-detection
git checkout develop
```

### 2. Create environment and install dependencies
```bash
conda create -n emotion-cnn python=3.10
conda activate emotion-cnn
pip install -r requirements.txt
```

### 3. Download the dataset
Download FER2013 from Kaggle and place it at:
```
/kaggle/input/datasets/msambare/fer2013
```
Or update `data_dir` in `configs/config.yaml` to your local path.

### 4. Train the model (Kaggle recommended)
```bash
python src/train.py
```
Checkpoint saved to `models/best_model.pt`.

### 5. Evaluate
```bash
python src/evaluate.py
```
Outputs classification report and confusion matrix to `docs/results/`.

### 6. Run the app
```bash
streamlit run app/streamlit_app.py
```

### 7. Run tests
```bash
pytest tests/ -v
```

---

## Configuration

All parameters are centralized in `configs/config.yaml`:

```yaml
data:
  data_dir: "/kaggle/input/datasets/msambare/fer2013"
  num_classes: 7
  image_size: 48
  mean: 0.5077
  std: 0.2551

training:
  batch_size: 64
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10

model:
  architecture: "cnn_scratch"
  dropout: 0.5
```

---

*Built by Taha HOUMMADI & Alae RHAZOUANI — Academic ML project, April 2026*