"""
Very general utilities

"""

import pathlib
import importlib
from typing import Callable, Any
from functools import wraps

import yaml


def call_once(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to ensure a function is only called once

    :param func: function to be decorated
    :returns: the decorated function - does the same thing as the input function
    :raises RuntimeError: if the function has already been called in this process

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper.called:
            wrapper.called = True
            return func(*args, **kwargs)
        raise RuntimeError(f"{func.__name__} has already been called")

    wrapper.called = False
    return wrapper


def rootdir() -> pathlib.Path:
    """
    Root dir of this git repo

    """
    return pathlib.Path(__file__).parents[2]


@call_once
def userconf() -> dict:
    """
    Get the user configuration.

    Can only be called once - this is to guarantee that we don't accidentally
    have any dependencies on the user config file.
    Otherwise, there might be a mixture of explicit and implicit dependence
    on the contents of the config file, which might lead to obscure bugs
    or things happening that we didn't expect, such as in the hyperparameter
    tuning script.

    :returns: The user configuration
    :raises: RuntimeError if called more than once

    """
    with open(rootdir() / "userconf.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config() -> dict:
    """
    Get the global config

    :returns: The config

    """
    with open(rootdir() / "config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class(name: str) -> type:
    """
    Load a class from a module given a string.

    :param name: the name of the class to load. Should be in the format module.class,
                 where module can also contain "."s (e.g. module.submodule.class)
    :returns: the class object

    """
    module_path, class_name = name.rsplit(".", 1)

    module = importlib.import_module(module_path)

    return getattr(module, class_name)
