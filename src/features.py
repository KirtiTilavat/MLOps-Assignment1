"""Feature preprocessing shared by training and serving.

The preprocessor is bundled inside the saved model pipeline, so the API
receives raw patient values and applies exactly the same transformations
that were used during training.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Binary 0/1 columns need no encoding; multi-level nominal codes are one-hot encoded.
BINARY_FEATURES = ["sex", "fbs", "exang"]
NOMINAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c not in BINARY_FEATURES]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, pass binary flags through, one-hot encode nominal codes."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_FEATURES,
            ),
        ]
    )
