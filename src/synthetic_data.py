"""Generate a small synthetic dataset for a LiDAR + imagery fusion demo.

This creates engineered features, not raw point clouds or real images.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GEOMETRY_FEATURES = [
    "height_m",
    "width_m",
    "verticality",
    "roughness",
    "point_density",
]
APPEARANCE_FEATURES = [
    "greenness",
    "brightness",
    "texture",
]
LABELS = ["pole", "tree", "building", "ground"]


def make_dataset(n_per_class: int = 300, random_state: int = 42) -> pd.DataFrame:
    """Return a reproducible synthetic object-level feature dataset."""
    if n_per_class < 10:
        raise ValueError("n_per_class must be at least 10")

    rng = np.random.default_rng(random_state)
    rows = []

    # Means and standard deviations are illustrative, not measurements.
    profiles = {
        "pole": {
            "geometry": ([7.0, 0.65, 0.82, 0.28, 0.58], [2.4, 0.45, 0.18, 0.20, 0.22]),
            "appearance": ([0.30, 0.48, 0.35], [0.20, 0.20, 0.20]),
        },
        "tree": {
            "geometry": ([6.5, 3.2, 0.40, 0.60, 0.53], [2.5, 1.8, 0.25, 0.24, 0.23]),
            "appearance": ([0.68, 0.50, 0.60], [0.22, 0.20, 0.22]),
        },
        "building": {
            "geometry": ([5.0, 6.5, 0.65, 0.28, 0.65], [2.5, 2.5, 0.22, 0.18, 0.22]),
            "appearance": ([0.35, 0.58, 0.38], [0.22, 0.20, 0.20]),
        },
        "ground": {
            "geometry": ([0.7, 4.5, 0.15, 0.30, 0.42], [0.8, 2.5, 0.14, 0.20, 0.22]),
            "appearance": ([0.42, 0.55, 0.30], [0.24, 0.20, 0.20]),
        },
    }

    for label, profile in profiles.items():
        geom_mean, geom_std = profile["geometry"]
        app_mean, app_std = profile["appearance"]
        geom = rng.normal(geom_mean, geom_std, size=(n_per_class, len(GEOMETRY_FEATURES)))
        app = rng.normal(app_mean, app_std, size=(n_per_class, len(APPEARANCE_FEATURES)))

        # Keep bounded features in plausible ranges.
        geom[:, 2] = np.clip(geom[:, 2], 0.0, 1.0)  # verticality
        geom[:, 3] = np.clip(geom[:, 3], 0.0, 1.0)  # roughness
        geom[:, 4] = np.clip(geom[:, 4], 0.0, 1.0)  # point density
        app = np.clip(app, 0.0, 1.0)

        for g, a in zip(geom, app):
            row = dict(zip(GEOMETRY_FEATURES, g))
            row.update(dict(zip(APPEARANCE_FEATURES, a)))
            row["label"] = label
            rows.append(row)

    data = pd.DataFrame(rows)
    return data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


if __name__ == "__main__":
    df = make_dataset()
    print(df.head())
    print("\nClass counts:\n", df["label"].value_counts())
