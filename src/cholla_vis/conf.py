# this file is dedicated to defining logic for tracking configuration

import argparse
from collections import ChainMap
import dataclasses
import io
import os
import tomllib
from types import MappingProxyType
from typing import Any, ClassVar


def _careful_init(
    full_conf: dict[str, Any],
    table_name: str,
    klass: Any,
    defaults: dict[str, Any] | None = None
) -> Any:
    if defaults is None:
        defaults = {}
    opt_fallbacks = {field: None for field in getattr(klass, "_OPTIONAL_FIELDS", ())}

    effective_table = ChainMap(full_conf.get(table_name, {}), defaults, opt_fallbacks)

    known_names = set(f.name for f in dataclasses.fields(klass))
    if len(known_names.symmetric_difference(effective_table.keys())) != 0:

        for k in effective_table:
            if k not in known_names:
                raise ValueError(f"`{table_name}.{k}` isn't a known parameter")
        for k in known_names:
            if k not in effective_table:
                raise ValueError(f"`{table_name}.{k}` is missing!")
    return klass(**effective_table)

@dataclasses.dataclass(frozen=True)
class PathConf:
    simdata_prefix: str | None
    intermediate_data: str
    processed_data: str

    _OPTIONAL_FIELDS: ClassVar[tuple[str, ...]] = ("simdata_prefix",)


    @classmethod
    def create_from_full_conf(
        cls,
        full_conf: dict[str: Any],
        defaults: dict[str, Any] | None = None
    ):
        return _careful_init(full_conf, "path", cls, defaults=defaults)
 


@dataclasses.dataclass(frozen=True)
class SimProp:
    restart_base: str | None
    start_snap: int
    stop_snap: int

    _OPTIONAL_FIELDS: ClassVar[tuple[str, ...]] = ("restart_base",)

    def __post_init__(self):
        assert 0 <= self.start_snap < self.stop_snap
        if self.start_snap == 0 and self.restart_base is not None:
            raise AssertionError("can't have a restart base when start_snap is 0")
        elif self.start_snap != 0 and self.restart_base is None:
            raise AssertionError("must have a restart base when start_snap isn't 0")

@dataclasses.dataclass(frozen=True)
class SimPropConf:
    data: MappingProxyType[str, SimProp]

    def __post_init__(self):
        names = set(self.data.keys())
        for name, val in self.data.items():
            if val.restart_base is None:
                continue
            elif name == val.restart_base:
                raise ValueError(f"{name!r} can't list itself as restart_base")
            elif val.restart_base not in names:
                # maybe don't raise an error here?
                raise AssertionError(
                    f"{name!r} lists an unknown simulation, {val.restart_base} as a "
                    "restart base"
                )

    @classmethod
    def create_from_full_conf(cls, full_conf: dict[str: Any]):
        parsed_data = {}
        data = full_conf.get("sim-props")
        for key, value in data.items():
            assert isinstance(value, dict)
            parsed_data[key] = _careful_init(data, key, SimProp)
        return cls(MappingProxyType(parsed_data))


@dataclasses.dataclass(frozen=True)
class FullConf:
    path_conf: PathConf
    simprop_conf: SimPropConf

    @classmethod
    def create_from_full_conf(cls, full_conf: dict[str: Any]):
        return cls(
            path_conf = PathConf.create_from_full_conf(full_conf),
            simprop_conf = SimPropConf.create_from_full_conf(full_conf),
        )


def build_conf_parser() -> argparse.ArgumentParser:
    """Builds an argument parser designed for use in scripts

    To use these arguments alongside other arguments, you might initialize this
    parser first and use the parents kwarg when constructing the script's
    parser.
    """

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--conf",
        help = "path to the configuration file",
        type=argparse.FileType("rb")
    )
    # it would be nice to let us specify arbitrary parameters
    # -C path.to.param_a:str=val_a
    # -C path.to.param_b:int=val_b
    return p

def path_conf_from_file(file: os.PathLike | io.IOBase) -> PathConf:
    try:
        with open(file, "rb") as f:
            data = tomllib.load(f)
    except TypeError:
        data = tomllib.load(file)
    return PathConf.create_from_full_conf(data)

def path_conf_from_cli(args: argparse.Namespace) -> PathConf:
    if args.conf is None:
        raise RuntimeError("--conf not provided")
    return path_conf_from_file(args.conf)

def full_conf_from_file(file: os.PathLike | io.IOBase) -> FullConf:
    try:
        with open(file, "rb") as f:
            data = tomllib.load(f)
    except TypeError:
        data = tomllib.load(file)
    return FullConf.create_from_full_conf(data)

def full_conf_from_cli(args: argparse.Namespace) -> SimPropConf:
    if args.conf is None:
        raise RuntimeError("--conf not provided")
    return full_conf_from_file(args.conf)
