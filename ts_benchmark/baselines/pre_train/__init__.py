# -*- coding: utf-8 -*-
from importlib import import_module

__all__ = [
    "TimerModel",
    "Moment",
    "TinyTimeMixer",
]


def _optional_import(module_path: str, attr_name: str):
    try:
        module = import_module(module_path)
    except ImportError:
        return None
    return getattr(module, attr_name, None)


TimerModel = _optional_import(
    "ts_benchmark.baselines.pre_train.model.timer", "TimerModel"
)
Moment = _optional_import(
    "ts_benchmark.baselines.pre_train.model.moment", "Moment"
)
TinyTimeMixer = _optional_import(
    "ts_benchmark.baselines.pre_train.model.tinytimemixer", "TinyTimeMixer"
)
