from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synthetic_data import (  # noqa: E402
    APPEARANCE_FEATURES,
    GEOMETRY_FEATURES,
    LABELS,
    make_dataset,
)


def test_dataset_has_expected_columns_and_classes():
    df = make_dataset(n_per_class=10, random_state=7)
    assert set(GEOMETRY_FEATURES + APPEARANCE_FEATURES + ["label"]).issubset(df.columns)
    assert set(df["label"].unique()) == set(LABELS)
    assert len(df) == 40


def test_dataset_is_reproducible():
    first = make_dataset(n_per_class=10, random_state=3)
    second = make_dataset(n_per_class=10, random_state=3)
    assert first.equals(second)
