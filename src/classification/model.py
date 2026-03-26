"""Propaganda technique classification with XLM-RoBERTa-large."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

# SemEval-2020 Task 11 labels.
TECHNIQUE_LABELS: list[str] = [
    "Appeal_to_Authority",
    "Appeal_to_Fear-Prejudice",
    "Bandwagon",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration-Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling-Labeling",
    "Repetition",
    "Slogans",
    "Thought-terminating_Cliches",
    "Whataboutism-Straw_Men-Red_Herring",
]


class PropagandaClassifier:
    """Wrapper around a fine-tuned XLM-RoBERTa-large classifier."""

    def __init__(self, model_dir: Path | None = None) -> None:
        """Initialize classifier state and optional model loading."""
        self._model: Any = None
        self._tokenizer: Any = None
        self.labels: list[str] = TECHNIQUE_LABELS

        if model_dir is not None:
            self.load_model(model_dir)

    def load_model(self, model_dir: Path) -> None:
        """Load model weights and tokenizer from model_dir."""
        raise NotImplementedError(
            f"Model loading not yet implemented. Place fine-tuned "
            f"XLM-RoBERTa-large weights in {model_dir} and update "
            f"this method to load them via transformers.AutoModel."
        )

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """Return per-technique probabilities for each input text."""
        raise NotImplementedError(
            "Prediction requires a trained model. Run fine-tuning on "
            "the SemEval-2020 Task 11 dataset first, then call "
            "load_model() before predict()."
        )

    def evaluate(
        self,
        test_df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "labels",
    ) -> dict[str, float]:
        """Compute evaluation metrics on a labelled test set."""
        raise NotImplementedError(
            "Evaluation requires a trained model and an annotated test "
            "set.  Complete fine-tuning and annotation before calling "
            "evaluate()."
        )
