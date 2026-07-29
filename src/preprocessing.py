"""Text preprocessing utilities for SpamShield AI.

This module is responsible only for text cleaning, tokenization, stopword
removal, stemming, and optional lemmatization.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

LOGGER = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_PATTERN = re.compile(r"<.*?>")
_NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True)
class TextPreprocessor:
    """Reusable preprocessing pipeline for short-text NLP tasks."""

    language: str = "english"
    use_lemmatization: bool = False
    lowercase: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    remove_special_characters: bool = True
    remove_numbers: bool = True
    remove_stopwords: bool = True
    stopwords_override: set[str] | None = None
    stemmer: PorterStemmer = field(default_factory=PorterStemmer)
    lemmatizer: WordNetLemmatizer = field(default_factory=WordNetLemmatizer)

    def preprocess(self, text: str | None) -> str:
        """Clean and normalize a single text value.

        Args:
            text: Raw input text.

        Returns:
            A normalized string suitable for vectorization.
        """
        if text is None:
            return ""

        processed_text = html.unescape(str(text))
        if self.remove_html:
            processed_text = _HTML_PATTERN.sub(" ", processed_text)
        if self.remove_urls:
            processed_text = _URL_PATTERN.sub(" ", processed_text)
        if self.lowercase:
            processed_text = processed_text.lower()
        if self.remove_special_characters or self.remove_numbers:
            processed_text = _NON_ALPHA_PATTERN.sub(" ", processed_text)

        tokens = self.tokenize(processed_text)
        tokens = self._filter_stopwords(tokens)
        tokens = self._stem_or_lemmatize(tokens)
        return " ".join(tokens).strip()

    def preprocess_many(self, texts: Iterable[str | None]) -> list[str]:
        """Preprocess a sequence of text values."""
        return [self.preprocess(text) for text in texts]

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a cleaned string into words."""
        normalized = _MULTI_SPACE_PATTERN.sub(" ", text).strip()
        if not normalized:
            return []
        return normalized.split(" ")

    def _filter_stopwords(self, tokens: list[str]) -> list[str]:
        if not self.remove_stopwords:
            return tokens

        stop_words = self._get_stop_words()
        return [token for token in tokens if token and token not in stop_words]

    def _stem_or_lemmatize(self, tokens: list[str]) -> list[str]:
        if self.use_lemmatization:
            return [self._lemmatize(token) for token in tokens if token]
        return [self.stemmer.stem(token) for token in tokens if token]

    def _lemmatize(self, token: str) -> str:
        try:
            return self.lemmatizer.lemmatize(token)
        except LookupError:
            LOGGER.warning("NLTK wordnet corpus is unavailable; falling back to original token for %s", token)
            return token

    def _get_stop_words(self) -> frozenset[str]:
        override = self.stopwords_override
        if override is not None:
            return frozenset(override)
        return _get_nltk_stopwords(self.language)


@lru_cache(maxsize=8)
def _get_nltk_stopwords(language: str) -> frozenset[str]:
    try:
        return frozenset(stopwords.words(language))
    except LookupError:
        LOGGER.info("NLTK stopwords corpus not found; attempting download for %s", language)
        _ensure_nltk_resource("stopwords")
        try:
            return frozenset(stopwords.words(language))
        except LookupError:
            LOGGER.warning("Stopwords corpus could not be loaded; continuing without stopword removal")
            return frozenset()


def _ensure_nltk_resource(resource_name: str) -> None:
    """Download an NLTK resource if it is missing."""
    try:
        nltk.data.find(f"corpora/{resource_name}")
    except LookupError:
        nltk.download(resource_name, quiet=True)


_DEFAULT_PREPROCESSOR = TextPreprocessor()


def preprocess_text(text: str | None, *, use_lemmatization: bool = False) -> str:
    """Preprocess a single text value using the default pipeline."""
    preprocessor = TextPreprocessor(use_lemmatization=use_lemmatization)
    return preprocessor.preprocess(text)


def preprocess_many(texts: Iterable[str | None], *, use_lemmatization: bool = False) -> list[str]:
    """Preprocess multiple text values using the default pipeline."""
    preprocessor = TextPreprocessor(use_lemmatization=use_lemmatization)
    return preprocessor.preprocess_many(texts)


def build_corpus(messages: Iterable[str | None], *, use_lemmatization: bool = False) -> list[str]:
    """Convert raw messages into a cleaned corpus suitable for vectorization."""
    return preprocess_many(messages, use_lemmatization=use_lemmatization)


def clean_text(text: str | None) -> str:
    """Backward-compatible convenience wrapper for default preprocessing."""
    return _DEFAULT_PREPROCESSOR.preprocess(text)
