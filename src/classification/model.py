"""Propaganda technique classification using XLM-RoBERTa-large.

Fine-tuned on the SemEval-2020 Task 11 dataset for 14-class propaganda
technique detection.  Supports multilingual input (English, Russian, Greek)
via the XLM-RoBERTa-large backbone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

# SemEval-2020 Task 11 propaganda technique labels.
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
    """Wrapper around a fine-tuned XLM-RoBERTa-large propaganda classifier.

    Intended workflow::

        clf = PropagandaClassifier()
        clf.load_model(MODELS_DIR / "xlm-roberta-propaganda")
        predictions = clf.predict(["Some political text"])
        metrics = clf.evaluate(test_df)
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        """Initialise the classifier.

        Args:
            model_dir: Directory containing saved model weights and
                tokenizer files.  If provided, ``load_model`` is called
                automatically.
        """
        self._model: Any = None
        self._tokenizer: Any = None
        self.labels: list[str] = TECHNIQUE_LABELS

        if model_dir is not None:
            self.load_model(model_dir)

    def load_model(self, model_dir: Path) -> None:
        """Load fine-tuned weights and tokenizer from *model_dir*.

        Args:
            model_dir: Path to the saved Hugging Face model directory.

        Raises:
            NotImplementedError: Model training is not yet complete.
        """
        raise NotImplementedError(
            f"Model loading not yet implemented. Place fine-tuned "
            f"XLM-RoBERTa-large weights in {model_dir} and update "
            f"this method to load them via transformers.AutoModel."
        )

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """Return per-technique probabilities for each input text.

        Args:
            texts: List of raw text strings (any supported language).

        Returns:
            List of dicts mapping each technique label to its predicted
            probability.

        Raises:
            NotImplementedError: Model is not yet loaded / trained.
        """
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
        """Compute evaluation metrics on a labelled test set.

        Args:
            test_df: DataFrame with text and ground-truth label columns.
            text_col: Name of the column containing input text.
            label_col: Name of the column containing ground-truth labels.

        Returns:
            Dict with macro/micro F1, precision, recall, and per-class
            metrics.

        Raises:
            NotImplementedError: Model is not yet loaded / trained.
        """
        raise NotImplementedError(
            "Evaluation requires a trained model and an annotated test "
            "set.  Complete fine-tuning and annotation before calling "
            "evaluate()."
        )
