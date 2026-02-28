"""Propaganda classification — draft module.

TODO
----
* Integrate a fine-tuned transformer (e.g. multilingual BERT) for
  propaganda technique detection.
* Load model weights from ``models/`` directory.
* Expose a ``predict(texts: list[str]) -> list[dict]`` API that returns
  per-text label probabilities.

This file is a placeholder; the model and full corpus are not yet
available.
"""

from pathlib import Path

from src.config import MODELS_DIR


def predict(texts: list[str]) -> list[dict]:
    """Classify *texts* for propaganda techniques.

    .. warning::
        Not yet implemented — returns empty predictions.
    """
    raise NotImplementedError(
        "Model not yet trained. Place weights in "
        f"{MODELS_DIR} and update this module."
    )
