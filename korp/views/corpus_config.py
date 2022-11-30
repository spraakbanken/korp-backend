import glob
import json
import os
from pathlib import Path

import yaml
from flask import Blueprint
from flask import current_app as app
from pymemcache.exceptions import MemcacheError
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from korp import utils
from korp.memcached import memcached

bp = Blueprint("corpus_config", __name__)


@bp.route("/corpus_config", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def corpus_config(args):
    """Get corpus configuration for a given mode or list of corpora. To be used by the Korp frontend.

    If no mode or corpora are specified, the mode 'default' is used.
    """
    mode_name = args.get("mode", "default")
    corpora = utils.parse_corpora(args)
    cache_checksum = utils.get_hash((mode_name, sorted(corpora), app.config["LAB_MODE"]))

    # Try to fetch config from cache
    if args["cache"]:
        with memcached.get_client() as mc:
            result = mc.get("%s:corpus_config_%s" % (utils.cache_prefix(mc, config=True), cache_checksum))
        if result:
            if "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
            yield result
            return

    result = get_mode(mode_name, corpora, args["cache"]) or {}
    result["modes"] = get_modes(mode_name)

    # Save to cache
    if args["cache"]:
        with memcached.get_client() as mc:
            try:
                added = mc.add("%s:corpus_config_%s" % (utils.cache_prefix(mc, config=True), cache_checksum), result)
            except MemcacheError:
                pass
            else:
                if added and "debug" in args:
                    result.setdefault("DEBUG", {})
                    result["DEBUG"]["cache_saved"] = True

    if "debug" in args:
        result.setdefault("DEBUG", {})
        result["DEBUG"]["yaml_loader"] = SafeLoader.__name__

    yield result


def get_modes(mode_name=None):
    """Get all modes data.

    Args:
        mode_name: Name of current mode. A hidden mode will only be included if it is the current mode.
    """
    modes = []
    for mode_file in (Path(app.config["CORPUS_CONFIG_DIR"]) / "modes").glob("*.yaml"):
        with open(mode_file, "r", encoding="utf-8") as f:
            mode = yaml.load(f, Loader=SafeLoader)
            # Only include hidden modes when accessed directly
            if mode.get("hidden") and not mode_name == mode_file.stem:
                continue
            modes.append({
                "mode": mode_file.stem,
                "label": mode.get("label", mode_file.stem),
                "order": mode.get("order")
            })
    return [
        {k: m[k] for k in m if k not in "order"} for m in sorted(modes, key=lambda x: (x["order"] is None, x["order"]))
    ]


def get_mode(mode_name: str, corpora: list, cache: bool):
    """Build configuration structure for a given mode.

    Args:
        mode_name: Name of mode to get.
        corpora: Optionally specify which corpora to include.
        cache: Whether to use cache.
    """
    try:
        with open(os.path.join(app.config["CORPUS_CONFIG_DIR"], "modes", mode_name + ".yaml"), "r",
                  encoding="utf-8") as fp:
            mode = yaml.load(fp, Loader=SafeLoader)
    except FileNotFoundError:
        return

    attr_types = {
        "positional": "pos_attributes",
        "structural": "struct_attributes",
        "custom": "custom_attributes"
    }

    mode["corpora"] = {}  # All corpora in mode
    mode["attributes"] = {t: {} for t in attr_types.values()}  # Attributes referred to by corpora
    attribute_presets = {t: {} for t in attr_types.values()}  # Attribute presets
    hash_to_attr = {}
    warnings = set()

    def get_new_attr_name(name: str) -> str:
        """Create a unique name for attribute, to be used as identifier."""
        while name in hash_to_attr.values():
            name += "_"
        return name

    if corpora:
        corpus_files = []
        for c in corpora:
            path = ""
            if ":" in c:
                path, _, c = c.partition(":")
            file_path = Path(app.config["CORPUS_CONFIG_DIR"]) / "corpora" / path.lower() / f"{c.lower()}.yaml"
            if file_path.is_file():
                corpus_files.append(file_path)
            else:
                warnings.add(f"The corpus {c!r} does not exist, or does not have a config file.")
    else:
        corpus_files = glob.glob(os.path.join(app.config["CORPUS_CONFIG_DIR"], "corpora", "*.yaml"))

    # Go through all corpora to see if they are included in mode
    for corpus_file in corpus_files:
        # Load corpus config from cache if possible
        cached_corpus = None
        if cache:
            with memcached.get_client() as mc:
                cached_corpus = mc.get("%s:corpus_config_%s" % (utils.cache_prefix(mc, config=True),
                                                                os.path.basename(corpus_file)))
            if cached_corpus:
                corpus_def = cached_corpus

        if not cached_corpus:
            with open(corpus_file, "r", encoding="utf-8") as fp:
                corpus_def = yaml.load(fp, Loader=SafeLoader)
            # Save to cache
            if cache:
                with memcached.get_client() as mc:
                    try:
                        mc.add("%s:corpus_config_%s" % (utils.cache_prefix(mc, config=True),
                                                        os.path.basename(corpus_file)), corpus_def)
                    except MemcacheError:
                        pass

        corpus_id = corpus_def["id"]

        # Check if corpus is included in selected mode
        if corpora or mode_name in [m["name"] for m in corpus_def.get("mode", [])]:
            for attr_type_name, attr_type in attr_types.items():
                if attr_type in corpus_def:
                    to_delete = []
                    for i, attr in enumerate(corpus_def[attr_type]):
                        for attr_name, attr_val in attr.items():
                            # A reference to an attribute preset
                            if isinstance(attr_val, str) or isinstance(attr_val, dict) and "preset" in attr_val:
                                if isinstance(attr_val, str):
                                    preset_name = attr_val
                                    attr_hash = utils.get_hash((attr_name, attr_val, attr_type))
                                else:
                                    preset_name = attr_val["preset"]
                                    attr_hash = utils.get_hash(
                                        (attr_name, json.dumps(attr_val, sort_keys=True), attr_type))

                                if attr_hash in hash_to_attr:  # Preset already loaded and ready to use
                                    corpus_def[attr_type][i] = hash_to_attr[attr_hash]
                                else:
                                    if preset_name not in attribute_presets[attr_type]:  # Preset not loaded yet
                                        try:
                                            with open(os.path.join(app.config["CORPUS_CONFIG_DIR"], "attributes",
                                                                   attr_type_name, preset_name + ".yaml"),
                                                      encoding="utf-8") as f:
                                                attr_def = yaml.load(f, Loader=SafeLoader)
                                                if not attr_def:
                                                    warnings.add(f"Preset {preset_name!r} is empty.")
                                                    to_delete.append(i)
                                                    continue
                                                attribute_presets[attr_type][preset_name] = attr_def
                                        except FileNotFoundError:
                                            to_delete.append(i)
                                            warnings.add(f"Attribute preset {preset_name!r} in corpus {corpus_id!r} "
                                                         "does not exist.")
                                            continue
                                    attr_id = get_new_attr_name(preset_name)
                                    hash_to_attr[attr_hash] = attr_id
                                    mode["attributes"][attr_type][attr_id] = attribute_presets[attr_type][
                                        preset_name].copy()
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
            corpus_modes = [mode for mode in corpus_def.get("mode", []) if mode["name"] == mode_name]
            if corpus_modes:
                corpus_mode_settings = corpus_modes.pop()
            else:
                corpus_mode_settings = {}

            # Skip corpus if it should only appear in lab mode, and we're not in lab mode
            if app.config["LAB_MODE"] or not corpus_mode_settings.get("lab_only", False):
                # Remove some keys from corpus config, as they are only used to create the full configuration
                corpus = {k: v for k, v in corpus_def.items() if k not in ("mode",)}

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

    return mode


def _add_corpus_to_folder(folders: dict, target_folder: str, corpus: str) -> None:
    """Add corpus to target_folder in folders.

    target_folder is a path with . as separator.
    """
    if not (target_folder and folders):
        return
    target = {"subfolders": folders}
    parts = target_folder.split(".")
    for part in parts:
        target.setdefault("subfolders", {})
        target = target["subfolders"][part]
    target.setdefault("corpora", [])
    target["corpora"].append(corpus)


def _remove_empty_folders(mode) -> None:
    """Remove empty folders from mode."""

    def should_include(folder):
        """Recurseively check for content in this folder or its subfolders."""
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

