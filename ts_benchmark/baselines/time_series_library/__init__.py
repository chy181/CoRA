# -*- coding: utf-8 -*-
from importlib import import_module

__all__ = [
    "Autoformer",
    "Crossformer",
    "DLinear",
    "ETSformer",
    "FEDformer",
    "FiLM",
    "Informer",
    "iTransformer",
    "Koopa",
    "LightTS",
    "Linear",
    "MICN",
    "NLinear",
    "Nonstationary_Transformer",
    "PatchTST",
    "Pyraformer",
    "Reformer",
    "TimesNet",
    "Transformer",
    "Triformer",
    "TimeMixer",
]


def _optional_import(module_path: str, attr_name: str):
    try:
        module = import_module(module_path)
    except ImportError:
        return None
    return getattr(module, attr_name, None)


Autoformer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.Autoformer", "Autoformer"
)
Crossformer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.Crossformer", "Crossformer"
)
DLinear = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.DLinear", "DLinear"
)
ETSformer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.ETSformer", "ETSformer"
)
FEDformer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.FEDformer", "FEDformer"
)
FiLM = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.FiLM", "FiLM"
)
Informer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.Informer", "Informer"
)
iTransformer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.iTransformer", "iTransformer"
)
Koopa = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.Koopa", "Koopa"
)
LightTS = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.LightTS", "LightTS"
)
Linear = _optional_import(
    "ts_benchmark.baselines.time_series_library.patchs.Linear", "Linear"
)
MICN = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.MICN", "MICN"
)
NLinear = _optional_import(
    "ts_benchmark.baselines.time_series_library.patchs.NLinear", "NLinear"
)
Nonstationary_Transformer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer",
    "Nonstationary_Transformer",
)
PatchTST = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.PatchTST", "PatchTST"
)
Pyraformer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.Pyraformer", "Pyraformer"
)
Reformer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.Reformer", "Reformer"
)
TimesNet = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.TimesNet", "TimesNet"
)
Transformer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.Transformer", "Transformer"
)
Triformer = _optional_import(
    "ts_benchmark.baselines.time_series_library.patchs.Triformer", "Triformer"
)
TimeMixer = _optional_import(
    "ts_benchmark.baselines.time_series_library.models.TimeMixer", "TimeMixer"
)
