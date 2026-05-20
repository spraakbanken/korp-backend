"""Router for corpus configuration."""

import json
from collections.abc import AsyncIterator
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Annotated, Any, TypeAlias

import anyio
import yaml
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from korp.api import schemas
from korp.config import settings
from korp.memcached import CacheError, Memcached

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from fastapi import APIRouter, Query

from korp import caching, utils
from korp.dependencies import CtxDep
from korp.handler import api_handler, docs_response

router = APIRouter(tags=["Corpus Information"])

CORPUS_CONFIG_DESCRIPTION = """Return the corpus configuration used by the Korp frontend.

Builds a JSON configuration describing corpora, shared attribute definitions, available modes, optional folder/grouping
metadata, and configuration warnings.

If `corpus` is omitted, the route includes the corpora that belong to the selected `mode`. If `corpus` is provided, only
the specified corpora are included.

Hidden modes are omitted from the returned `modes` list unless the hidden mode is requested directly.

Any extra fields in the mode or corpus configuration YAML files are included in the output as-is.
"""

Label: TypeAlias = str | dict[str, str]


CorpusParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "Comma-separated list of corpora to include in the configuration. If specified, this overrides the "
            "corpus list from `mode`."
        ),
        examples=[["ROMI,SUC3"]],
    ),
    BeforeValidator(utils.split_csv),
    AfterValidator(lambda v: [x.upper() for x in v]),
]


class ModeSummary(BaseModel):
    """Mode entry returned in the `modes` list."""

    mode: str = Field(..., description="Mode id.", examples=["default"])
    label: Label = Field(
        ...,
        description=(
            "Human-readable mode label. Either a simple string or a dictionary mapping language codes to labels."
        ),
        examples=[{"eng": "Default"}],
    )


class CorpusConfigResponse(schemas.CommonResponse):
    """Response model for `/corpus_config` route."""

    model_config = ConfigDict(extra="allow")

    label: Label | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Human-readable label for the selected mode, if defined by the mode file. Either a simple string or a "
            "dictionary mapping language codes to labels."
        ),
        examples=[{"eng": "Default", "swe": "Standard"}],
    )
    corpora: dict[str, dict[str, Any]] = Field(
        ...,
        description=(
            "Corpus configuration keyed by corpus id. The exact fields are defined by the corpus configuration YAML; "
            "common fields include `id`, `description`, `pos_attributes`, and `struct_attributes`."
        ),
    )
    attributes: dict[str, dict[str, dict[str, Any]]] | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Shared attribute definitions grouped by attribute kind, for example `pos_attributes`, "
            "`struct_attributes`, and `custom_attributes`."
        ),
    )
    modes: list[ModeSummary] = Field(
        ...,
        description="Modes available to the frontend. Hidden modes are included only when requested directly.",
    )
    folders: dict[str, Any] | SkipJsonSchema[None] = Field(
        None,
        description="Optional folder tree for grouping corpora in the frontend.",
    )
    preselected_corpora: list[str] | SkipJsonSchema[None] = Field(
        None,
        description="Optional corpora that should be selected by default in the frontend.",
        examples=[["ROMI", "SUC3"]],
    )
    warnings: list[str] | SkipJsonSchema[None] = Field(
        None,
        description="Configuration warnings, such as missing attribute presets or missing corpus config files.",
    )


@router.get(
    "/corpus_config",
    response_model=None,
    responses=docs_response(CorpusConfigResponse),
    name="Corpus Configuration",
    summary="Corpus Configuration",
    description=CORPUS_CONFIG_DESCRIPTION,
)
@router.post("/corpus_config", response_model=None, include_in_schema=False)
@api_handler
async def corpus_config(
    ctx: CtxDep,
    mode: Annotated[
        str,
        Query(
            description="Mode to build configuration for. Defaults to `default`.",
            examples=["default"],
        ),
    ] = "default",
    corpus: CorpusParam = None,
) -> AsyncIterator[dict]:
    """Get corpus configuration for a given mode or list of corpora. To be used by the Korp frontend.

    If no mode or corpora are specified, the mode 'default' is used.

    Args:
        ctx: Request context.
        mode: Mode to get configuration for.
        corpus: Comma-separated list of corpora to include in configuration. If specified, overrides the mode's corpus
            list.

    Yields:
        Corpus configuration structure.

    Raises:
        NameError: If the specified mode does not exist.
        RuntimeError: If corpus configuration directory is not set in settings.
    """
    corpora = corpus or []
    cache_checksum = utils.get_hash((mode, sorted(corpora), settings.LAB_MODE))
    cache = ctx.cache

    # Try to fetch complete config from cache
    if ctx.common.cache:
        result = await cache.get(f"{await caching.cache_prefix(cache, config=True)}:corpus_config_{cache_checksum}")
        if result:
            if ctx.common.debug:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
            yield result
            return

    if not settings.CORPUS_CONFIG_DIR:
        raise RuntimeError("Corpus config directory is not set in settings, cannot fetch corpus configuration.")

    result = await get_mode(mode, corpora, cache if ctx.common.cache else None)
    if result is None:
        raise NameError(f"The mode {mode!r} does not exist.")
    result["modes"] = get_modes(mode)

    # Save to cache
    if ctx.common.cache:
        try:
            added = await cache.add(
                f"{await caching.cache_prefix(cache, config=True)}:corpus_config_{cache_checksum}", result
            )
        except CacheError:
            pass
        else:
            if added and ctx.common.debug:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_saved"] = True

    if ctx.common.debug:
        result.setdefault("DEBUG", {})
        result["DEBUG"]["yaml_loader"] = SafeLoader.__name__

    yield result


def get_modes(current_mode: str | None = None) -> list[dict]:
    """Get all modes data.

    Args:
        current_mode: Name of current mode. A hidden mode will only be included if it is the current mode.

    Returns:
        List of modes with their basic information.
    """
    assert settings.CORPUS_CONFIG_DIR
    modes = []
    for mode_file in (Path(settings.CORPUS_CONFIG_DIR) / "modes").glob("*.yaml"):
        with mode_file.open("r", encoding="utf-8") as f:
            mode = yaml.load(f, Loader=SafeLoader)
            # Only include hidden modes when accessed directly
            if mode.get("hidden") and current_mode != mode_file.stem:
                continue
            modes.append(
                {"mode": mode_file.stem, "label": mode.get("label", mode_file.stem), "order": mode.get("order")}
            )
    return [
        {k: v for k, v in m.items() if k != "order"}
        for m in sorted(modes, key=lambda x: (x["order"] is None, x["order"]))
    ]


def _get_mode_sync(
    mode: dict,
    mode_name: str,
    corpora: list,
    corpus_files: list[Path],
    cached_corpora: dict[Path, dict] | None = None,
) -> dict[str, dict]:
    """Build configuration structure for a given mode (synchronous part).

    Args:
        mode: Mode configuration structure to populate.
        mode_name: Name of mode to get.
        corpora: Optionally specify which corpora to include, otherwise all corpora in mode are included.
        corpus_files: Iterator of corpus config file paths.
        cached_corpora: Cached corpus configurations, if available. None if not using cache.

    Returns:
        Dictionary of corpus configurations to be saved to cache.
    """
    assert settings.CORPUS_CONFIG_DIR
    attr_types = {"positional": "pos_attributes", "structural": "struct_attributes", "custom": "custom_attributes"}

    mode["corpora"] = {}  # All corpora in mode
    mode["attributes"] = {t: {} for t in attr_types.values()}  # Attributes referred to by corpora
    attribute_presets = {t: {} for t in attr_types.values()}  # Attribute presets
    hash_to_attr = {}
    used_attr_names: set[str] = set()
    warnings = set()

    def get_new_attr_name(name: str) -> str:
        """Create a unique name for attribute, to be used as identifier.

        Args:
            name: Proposed name for attribute.

        Returns:
            Unique name for attribute.
        """
        while name in used_attr_names:
            name += "_"
        used_attr_names.add(name)
        return name

    save_to_cache = {}

    # Go through all corpora to see if they are included in mode
    for corpus_file in corpus_files:
        # Load corpus config from cache if possible
        corpus_def = None
        if cached_corpora and (cached_corpus := cached_corpora.get(corpus_file)):
            corpus_def = cached_corpus

        if not corpus_def:
            with corpus_file.open("r", encoding="utf-8") as fp:
                corpus_def = yaml.load(fp, Loader=SafeLoader)

            if cached_corpora is not None:
                save_to_cache[corpus_file] = deepcopy(corpus_def)

        corpus_id = corpus_def["id"]

        # Skip corpus if it's not included in the selected mode, unless specific corpora are requested
        if not corpora and not any(m["name"] == mode_name for m in corpus_def.get("mode", [])):
            continue
        for attr_type_name, attr_type in attr_types.items():
            if attr_type in corpus_def:
                to_delete = []
                for i, attr in enumerate(corpus_def[attr_type]):
                    for attr_name, attr_val in attr.items():
                        # A reference to an attribute preset
                        if isinstance(attr_val, str) or (isinstance(attr_val, dict) and "preset" in attr_val):
                            if isinstance(attr_val, str):
                                preset_name = attr_val
                                attr_hash = utils.get_hash((attr_name, attr_val, attr_type))
                            else:
                                preset_name = attr_val["preset"]
                                attr_hash = utils.get_hash((attr_name, json.dumps(attr_val, sort_keys=True), attr_type))

                            if attr_hash in hash_to_attr:  # Preset already loaded and ready to use
                                corpus_def[attr_type][i] = hash_to_attr[attr_hash]
                            else:
                                if preset_name not in attribute_presets[attr_type]:  # Preset not loaded yet
                                    try:
                                        with Path(
                                            settings.CORPUS_CONFIG_DIR,
                                            "attributes",
                                            attr_type_name,
                                            preset_name + ".yaml",
                                        ).open(encoding="utf-8") as f:
                                            attr_def = yaml.load(f, Loader=SafeLoader)
                                            if not attr_def:
                                                warnings.add(f"Preset {preset_name!r} is empty.")
                                                to_delete.append(i)
                                                continue
                                            attribute_presets[attr_type][preset_name] = attr_def
                                    except FileNotFoundError:
                                        to_delete.append(i)
                                        warnings.add(
                                            f"Attribute preset {preset_name!r} in corpus {corpus_id!r} does not exist."
                                        )
                                        continue
                                attr_id = get_new_attr_name(preset_name)
                                hash_to_attr[attr_hash] = attr_id
                                mode["attributes"][attr_type][attr_id] = attribute_presets[attr_type][
                                    preset_name
                                ].copy()
                                mode["attributes"][attr_type][attr_id].update({"name": attr_name})
                                if isinstance(attr_val, dict):
                                    # Override preset values
                                    del attr_val["preset"]
                                    mode["attributes"][attr_type][attr_id].update(attr_val)
                                corpus_def[attr_type][i] = attr_id

                        # Inline attribute definition
                        elif isinstance(attr_val, dict):
                            attr_hash = utils.get_hash((attr_name, json.dumps(attr_val, sort_keys=True), attr_type))
                            if attr_hash in hash_to_attr:  # Identical attribute has previously been used
                                corpus_def[attr_type][i] = hash_to_attr[attr_hash]
                            else:
                                attr_id = get_new_attr_name(attr_name)
                                hash_to_attr[attr_hash] = attr_id
                                attr_val.update({"name": attr_name})
                                mode["attributes"][attr_type][attr_id] = attr_val
                                corpus_def[attr_type][i] = attr_id
                for i in reversed(to_delete):
                    del corpus_def[attr_type][i]
        corpus_modes = [m for m in corpus_def.get("mode", []) if m["name"] == mode_name]
        corpus_mode_settings = corpus_modes.pop() if corpus_modes else {}

        # Skip corpus if it should only appear in lab mode, and we're not in lab mode
        if settings.LAB_MODE or not corpus_mode_settings.get("lab_only", False):
            # Remove some keys from corpus config, as they are only used to create the full configuration
            corpus = {k: v for k, v in corpus_def.items() if k != "mode"}

            folders = corpus_mode_settings.get("folder", [])
            if not isinstance(folders, list):
                folders = [folders]
            for folder in folders:
                try:
                    _add_corpus_to_folder(mode.get("folders"), folder, corpus_id)
                except KeyError:
                    warnings.add(f"The folder '{folder}' referred to by the corpus '{corpus_id}' doesn't exist.")

            # Add corpus configuration to mode
            mode["corpora"][corpus_id] = corpus

    if corpora and "preselected_corpora" in mode:
        del mode["preselected_corpora"]

    _remove_empty_folders(mode)
    if warnings:
        mode["warnings"] = list(warnings)

    return save_to_cache


async def get_mode(mode_name: str, corpora: list, cache: Memcached | None = None) -> dict | None:
    """Build configuration structure for a given mode.

    Args:
        mode_name: Name of mode to get.
        corpora: Optionally specify which corpora to include, otherwise all corpora in mode are included.
        cache: Memcached instance to use for caching.

    Returns:
        Mode configuration structure, or None if mode does not exist.
    """
    assert settings.CORPUS_CONFIG_DIR
    warnings = set()
    try:
        with Path(settings.CORPUS_CONFIG_DIR, "modes", mode_name + ".yaml").open("r", encoding="utf-8") as fp:
            mode = yaml.load(fp, Loader=SafeLoader)
    except FileNotFoundError:
        return None

    if corpora:
        corpus_files = []
        for c in corpora:
            file_path = Path(settings.CORPUS_CONFIG_DIR) / "corpora" / f"{c.lower()}.yaml"
            if file_path.is_file():
                corpus_files.append(file_path)
            else:
                warnings.add(f"The corpus {c!r} does not exist, or does not have a config file.")
    else:
        corpus_files = list(Path(settings.CORPUS_CONFIG_DIR, "corpora").glob("*.yaml"))

    cached_corpora: dict[Path, dict] | None = None
    cache_prefix = None

    if cache:
        cache_prefix = await caching.cache_prefix(cache, config=True)
        cache_keys = {
            f"{cache_prefix}:corpus_config_{Path(corpus_file).name}": corpus_file for corpus_file in corpus_files
        }
        cached_corpora = {
            cache_keys[corpus_key]: data for corpus_key, data in (await cache.get_many(cache_keys.keys())).items()
        }

    to_cache = await anyio.to_thread.run_sync(  # type: ignore
        partial(_get_mode_sync, mode, mode_name, corpora, corpus_files, cached_corpora)
    )

    if cache:
        for corpus_file, corpus_def in to_cache.items():
            try:
                await cache.add(
                    f"{cache_prefix}:corpus_config_{Path(corpus_file).name}",
                    corpus_def,
                )
            except CacheError:
                pass

    if warnings:
        mode.setdefault("warnings", [])
        mode["warnings"].extend(warnings)

    return mode


def _add_corpus_to_folder(folders: dict | None, target_folder: str, corpus: str) -> None:
    """Add corpus to target_folder in folders.

    target_folder is a path with . as separator.
    """
    if not (target_folder and folders):
        return
    target: dict = {"subfolders": folders}
    parts = target_folder.split(".")
    for part in parts:
        target.setdefault("subfolders", {})
        target = target["subfolders"][part]
    target.setdefault("corpora", [])
    target["corpora"].append(corpus)


def _remove_empty_folders(mode: dict) -> None:
    """Remove empty folders from mode."""

    def should_include(folder: dict) -> bool:
        """Recurseively check for content in this folder or its subfolders.

        Args:
            folder: Folder to check.

        Returns:
            True if folder or any of its subfolders contain corpora, False otherwise.
        """
        include = "corpora" in folder

        for subfolder_name, subfolder in list(folder.get("subfolders", {}).items()):
            include_subfolder = should_include(subfolder)
            if not include_subfolder:
                del folder["subfolders"][subfolder_name]
            if not include:
                # If current folder has no content but one of its subfolder has, it should be included
                include = include_subfolder
        return include

    mode_folders = mode.get("folders", {})
    for folder_id, f in list(mode_folders.items()):
        if not should_include(f):
            del mode_folders[folder_id]
