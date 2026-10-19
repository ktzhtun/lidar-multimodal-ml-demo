"""Train and compare geometry-only and multimodal baseline classifiers."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

# Support running as `python src/train.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_data import (  # noqa: E402
    APPEARANCE_FEATURES,
    GEOMETRY_FEATURES,
    make_dataset,
)


def evaluate_model(name, features, train_df, test_df):
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(train_df[features], train_df["label"])
    predictions = model.predict(test_df[features])

    accuracy = accuracy_score(test_df["label"], predictions)
    weighted_f1 = f1_score(test_df["label"], predictions, average="weighted")
    print(f"\n{'=' * 65}\n{name}")
    print(f"Accuracy:   {accuracy:.3f}")
    print(f"Weighted F1:{weighted_f1: .3f}")
    print("\nClassification report:")
    print(classification_report(test_df["label"], predictions, zero_division=0))
    return model, predictions, accuracy, weighted_f1


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    data = make_dataset(n_per_class=300, random_state=42)
    train_df, test_df = train_test_split(
        data,
        test_size=0.25,
        random_state=42,
        stratify=data["label"],
    )

    geometry_model, geometry_pred, geometry_acc, geometry_f1 = evaluate_model(
        "Baseline: geometry features only",
        GEOMETRY_FEATURES,
        train_df,
        test_df,
    )
    fusion_features = GEOMETRY_FEATURES + APPEARANCE_FEATURES
    fusion_model, fusion_pred, fusion_acc, fusion_f1 = evaluate_model(
        "Early fusion: geometry + appearance features",
        fusion_features,
        train_df,
        test_df,
    )

    summary = pd.DataFrame([
        {"model": "geometry_only", "accuracy": geometry_acc, "weighted_f1": geometry_f1},
        {"model": "early_fusion", "accuracy": fusion_acc, "weighted_f1": fusion_f1},
    ])
    summary.to_csv(output_dir / "metrics.csv", index=False)
    print("\nSummary:\n", summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        test_df["label"], geometry_pred, ax=axes[0], colorbar=False
    )
    axes[0].set_title("Geometry only")
    ConfusionMatrixDisplay.from_predictions(
        test_df["label"], fusion_pred, ax=axes[1], colorbar=False
    )
    axes[1].set_title("Geometry + appearance")
    fig.suptitle("Synthetic object classification — held-out test set")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved metrics and confusion matrices to: {output_dir.resolve()}")
    print("\nNote: synthetic educational data only; these metrics do not represent real-world performance.")


if __name__ == "__main__":
    main()
