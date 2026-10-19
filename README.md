# LiDAR + Aerial Imagery: A Tiny Multimodal ML Demo

A project demonstrating how two data sources can be combined to classify simplified infrastructure-scene objects.

**Important:** This project uses **synthetic features**, not real LiDAR scans or aerial photographs. It is an educational proof of concept for multimodal feature fusion, not a production-ready infrastructure model.

## What it demonstrates

- How LiDAR-like geometry features and image-like appearance features differ
- A baseline model using geometry features alone
- A second model using geometry + appearance features (early feature fusion)
- Evaluation with a classification report and confusion matrix
- Reproducible experiments using a fixed random seed

## Example task

Classify each detected object as one of:

- `pole`
- `tree`
- `building`
- `ground`

Each sample represents an already-cropped object. This deliberately keeps the project small: it does **not** solve object detection or point-cloud segmentation.

## Feature design

| Modality              | Example features                                     | What they approximate                        |
| --------------------- | ---------------------------------------------------- | -------------------------------------------- |
| LiDAR-like geometry   | height, width, verticality, roughness, point density | Shape and spatial structure                  |
| Image-like appearance | greenness, brightness, texture                       | Visual appearance from an aligned image crop |

In a real system, features from both sources would need to be spatially aligned and quality-checked before fusion.

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

The script prints test-set results for both models and saves a confusion-matrix plot to `outputs/`.

## Project structure

```text
neara-multimodal-ml-demo/
├── README.md
├── requirements.txt
├── src/
│   ├── synthetic_data.py
│   └── train.py
├── tests/
│   └── test_synthetic_data.py
└── outputs/                    # created when you run training
```

## How the experiment works

1. Generate a reproducible synthetic dataset.
2. Split it into training and test sets.
3. Train a Random Forest using geometry features only.
4. Train another Random Forest using geometry and appearance features.
5. Compare accuracy and weighted F1 on the same held-out test set.

The synthetic distributions are intentionally imperfect, so the experiment is useful for practising the pipeline. Results are not evidence of real-world performance, and feature fusion is not guaranteed to improve metrics on every run or dataset.

```bash
pytest
```

## Suggested next improvements

1. Add a `--seed` argument and compare results across several seeds.
2. Add missing-modality experiments (e.g. imagery unavailable).
3. Try a simple neural network and compare it to Random Forest.
4. Replace synthetic data with a small, appropriately licensed public dataset.
