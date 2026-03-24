# src/analysis/topic_modeling.py
"""BERTopic narrative clustering — multilingual (EN / RU / EL).

Each language is modelled independently with its own sentence-transformer
embedding and vocabulary, then outputs are merged into a single CSV with
a `lang` column for downstream cross-language analysis.

Usage (from main.ipynb or CLI):
    from src.analysis.topic_modeling import run_all_languages
    results = run_all_languages()
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance

from src.config import (
    CYRILLIC_LEMMATIZED_CSV,
    LATIN_LEMMATIZED_CSV,
    GREEK_LEMMATIZED_CSV,
    BERTOPIC_MODEL_DIR,
    BERTOPIC_TOPICS_CSV,
    BERTOPIC_TOPICINFO_CSV,
    ROOT_DIR,
)

logger = logging.getLogger(__name__)
_CONFIG_FILE = ROOT_DIR / "configs" / "bertopic.yaml"

# Maps lang tag → (lemmatized CSV path, text column, lemma column)
_LANG_SOURCES: dict[str, tuple[Path, str, str]] = {
    "en": (LATIN_LEMMATIZED_CSV,    "text_cleaned", "lemmas"),
    "ru": (CYRILLIC_LEMMATIZED_CSV, "text_cleaned", "lemmas"),
    "el": (GREEK_LEMMATIZED_CSV,    "text_cleaned", "lemmas"),
}


def _load_config() -> dict:
    with open(_CONFIG_FILE) as fh:
        return yaml.safe_load(fh)


def _vocab_from_lemmas(series: pd.Series) -> list[str] | None:
    """Build a flat vocabulary list from a column of lemma lists."""
    vocab: set[str] = set()
    for cell in series.dropna():
        parsed = ast.literal_eval(cell) if isinstance(cell, str) else cell
        if isinstance(parsed, list):
            vocab.update(parsed)
    return sorted(vocab) if vocab else None


def _build_model(cfg: dict, lang: str, vocab: list[str] | None) -> BERTopic:
    lang_cfg  = cfg["languages"][lang]
    ec, uc    = cfg["embedding"], cfg["umap"]
    hc, vc    = cfg["hdbscan"],   cfg["vectorizer"]
    rc        = cfg["representation"]
    seed      = cfg["random_seed"]

    embedding_model = SentenceTransformer(
        lang_cfg["embedding_model"], device=ec["device"]
    )
    umap_model = UMAP(
        n_neighbors=uc["n_neighbors"],
        n_components=uc["n_components"],
        min_dist=uc["min_dist"],
        metric=uc["metric"],
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=hc["min_cluster_size"],
        min_samples=hc["min_samples"],
        metric=hc["metric"],
        cluster_selection_method=hc["cluster_selection_method"],
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        vocabulary=vocab,                            # lemma-constrained vocab
        ngram_range=tuple(vc["ngram_range"]),
        min_df=vc["min_df"],
        max_features=vc.get("max_features") if vocab is None else None,
        stop_words=lang_cfg["vectorizer_stop_words"],
    )
    representation = MaximalMarginalRelevance(diversity=rc["diversity"])

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation,
        top_n_words=rc["top_n_words"],
        nr_topics=cfg.get("nr_topics", "auto"),
        calculate_probabilities=False,
        verbose=True,
    )


def run_language(lang: str, cfg: dict | None = None) -> pd.DataFrame:
    """Fit BERTopic for one language. Returns df with topic columns added."""
    if cfg is None:
        cfg = _load_config()

    csv_path, text_col, lemma_col = _LANG_SOURCES[lang]
    if not Path(csv_path).exists():
        logger.warning("Skipping %s — file not found: %s", lang, csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("Skipping %s — empty dataframe.", lang)
        return pd.DataFrame()

    docs  = df[text_col].fillna("").tolist()
    vocab = _vocab_from_lemmas(df[lemma_col]) if lemma_col in df.columns else None

    logger.info("[%s] Building model — %d documents, vocab size: %s",
                lang, len(docs), len(vocab) if vocab else "auto")

    model  = _build_model(cfg, lang, vocab)
    topics, _ = model.fit_transform(docs)

    df = df.copy()
    df["topic_id"]  = topics
    df["lang"]      = lang

    topic_info = model.get_topic_info()
    label_map  = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df["topic_label"] = df["topic_id"].map(label_map)

    # Persist model artefacts per language
    save_path = BERTOPIC_MODEL_DIR / f"bertopic_{lang}"
    model.save(
        str(save_path),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=cfg["languages"][lang]["embedding_model"],
    )
    logger.info("[%s] Model saved → %s", lang, save_path)

    # Per-language topic info
    lang_info_path = BERTOPIC_TOPICINFO_CSV.parent / f"bertopic_topic_info_{lang}.csv"
    topic_info.to_csv(lang_info_path, index=False)
    logger.info("[%s] %d topics found (excl. noise).",
                lang, (topic_info["Topic"] != -1).sum())

    return df


def run_all_languages(langs: list[str] | None = None) -> pd.DataFrame:
    """Run BERTopic for EN, RU, EL and merge into one output CSV.

    Args:
        langs: subset to run, e.g. ["en", "ru"]. Defaults to all three.

    Returns:
        Merged DataFrame with columns:
        message_id, date, channel, lang, topic_id, topic_label
    """
    if langs is None:
        langs = ["en", "ru", "el"]

    cfg     = _load_config()
    results = []

    for lang in langs:
        logger.info("=" * 60)
        logger.info("Starting BERTopic run: %s", lang.upper())
        lang_df = run_language(lang, cfg)
        if not lang_df.empty:
            results.append(lang_df)

    if not results:
        raise RuntimeError("All language runs produced empty results — check CSV paths.")

    keep_cols = [c for c in
                 ["message_id", "date", "channel", "lang", "topic_id", "topic_label"]
                 if c in results[0].columns]

    merged = pd.concat([r[keep_cols] for r in results], ignore_index=True)
    merged.to_csv(BERTOPIC_TOPICS_CSV, index=False)
    logger.info("Merged topics saved → %s  (%d rows)", BERTOPIC_TOPICS_CSV, len(merged))

    return merged
