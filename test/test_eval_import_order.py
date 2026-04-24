"""Regression tests for lazy import behavior in evaluation entrypoints."""

import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_MODULES = [
    "eval_arcc",
    "eval_gsm8k",
    "eval_math",
    "eval_mmlust",
    "eval_rag",
]


def _pop_modules(*prefixes: str) -> dict[str, object]:
    """Remove matching modules from `sys.modules` and return them for restoration."""
    removed = {}
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            removed[name] = sys.modules.pop(name)
    return removed


def _install_eval_import_stubs() -> None:
    """Install minimal dependency stubs so import-order tests stay lightweight."""
    datasets_module = types.ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: None

    class _Dataset:
        @classmethod
        def from_dict(cls, data):
            return data

        @classmethod
        def load_from_disk(cls, path):
            return path

    datasets_module.Dataset = _Dataset

    math_verify_module = types.ModuleType("math_verify")
    math_verify_module.parse = lambda value: value
    math_verify_module.verify = lambda left, right: left == right

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda *args, **kwargs: None

    sys.modules["datasets"] = datasets_module
    sys.modules["math_verify"] = math_verify_module
    sys.modules["tqdm"] = tqdm_module


@pytest.mark.parametrize("module_name", EVAL_MODULES)
def test_eval_module_import_does_not_eagerly_import_transformers(module_name):
    removed = _pop_modules(
        "datasets",
        "math_verify",
        "tqdm",
        "transformers",
        "utils",
        module_name,
    )
    try:
        _install_eval_import_stubs()

        importlib.import_module(module_name)

        assert "transformers" not in sys.modules
    finally:
        _pop_modules(
            "datasets",
            "math_verify",
            "tqdm",
            "transformers",
            "utils",
            module_name,
        )
        sys.modules.update(removed)
