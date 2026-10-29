"""Corpus configuration file utilities."""

from pathlib import Path

from korp import utils
from korp.config import settings


def get_corpus_config_files() -> dict[str, Path]:
    """Find all corpus configuration files in the configured directory.

    Returns:
        Configuration paths keyed by normalized corpus ID.

    Raises:
        ValueError: If filenames are not unique after case normalization.
    """
    if not settings.CORPUS_CONFIG_DIR:
        return {}
    files: dict[str, Path] = {}
    directory = Path(settings.CORPUS_CONFIG_DIR, "corpora")
    for path in sorted(directory.rglob("*.yaml"), key=lambda p: (p.name, str(p))):
        if not path.is_file():
            continue
        name = utils.normalize_corpus_id(path.stem)
        if name in files:
            raise ValueError(f"Duplicate corpus config filename {name + '.yaml'!r}: {files[name]} and {path}.")
        files[name] = path
    return files
