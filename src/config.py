"""Application configuration for SpamShield AI.

This module centralizes filesystem paths and lightweight runtime settings so
other modules can stay focused on preprocessing, feature extraction, training,
prediction, and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_NAME: Final[str] = "SpamShield AI"
DEFAULT_DATASET_FILENAME: Final[str] = "SMSSpamCollection"
DEFAULT_TEST_SIZE: Final[float] = 0.2
DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_MAX_FEATURES: Final[int] = 2500
DEFAULT_NGRAM_RANGE: Final[tuple[int, int]] = (1, 2)
DEFAULT_OUTPUT_LABEL_SPAM: Final[str] = "spam"
DEFAULT_OUTPUT_LABEL_HAM: Final[str] = "ham"


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Canonical filesystem locations used by the application."""

    root: Path
    assets_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    models_dir: Path
    vectorizers_dir: Path
    notebooks_dir: Path
    pages_dir: Path
    src_dir: Path

    @classmethod
    def from_project_root(cls, root: Path | None = None) -> "ProjectPaths":
        """Create a path bundle from the project root.

        Args:
            root: Optional override for the repository root.

        Returns:
            A normalized ``ProjectPaths`` instance.
        """
        project_root = root or Path(__file__).resolve().parents[1]
        data_dir = project_root / "data"
        return cls(
            root=project_root,
            assets_dir=project_root / "assets",
            data_dir=data_dir,
            raw_data_dir=data_dir / "raw",
            processed_data_dir=data_dir / "processed",
            models_dir=project_root / "models",
            vectorizers_dir=project_root / "vectorizers",
            notebooks_dir=project_root / "notebooks",
            pages_dir=project_root / "pages",
            src_dir=project_root / "src",
        )


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime configuration for the Streamlit application and ML pipeline."""

    project_name: str = PROJECT_NAME
    dataset_filename: str = DEFAULT_DATASET_FILENAME
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    max_features: int = DEFAULT_MAX_FEATURES
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE
    spam_label: str = DEFAULT_OUTPUT_LABEL_SPAM
    ham_label: str = DEFAULT_OUTPUT_LABEL_HAM
    paths: ProjectPaths = ProjectPaths.from_project_root()

    @property
    def dataset_path(self) -> Path:
        """Path to the raw SMS spam dataset."""
        return self.paths.raw_data_dir / self.dataset_filename

    def model_path(self, filename: str) -> Path:
        """Return the absolute path for a persisted model artifact."""
        return self.paths.models_dir / filename

    def vectorizer_path(self, filename: str) -> Path:
        """Return the absolute path for a persisted vectorizer artifact."""
        return self.paths.vectorizers_dir / filename


def get_config() -> AppConfig:
    """Return the default application configuration."""
    return AppConfig()


def get_paths() -> ProjectPaths:
    """Return the canonical project paths."""
    return ProjectPaths.from_project_root()
