# this file is dedicated to defining logic for tracking configuration

import argparse
from collections import ChainMap
import dataclasses
import io
import os
import tomllib

@dataclasses.dataclass(frozen=True)
class PathConf:
    simdata_prefix: str | None
    intermediate_data: str
    processed_data: str

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

def _careful_init(full_conf, table_name, klass, defaults = None):
    if defaults is None:
        defaults = {}

    effective_table = ChainMap(full_conf.get(table_name, {}), defaults)

    known_names = set(f.name for f in dataclasses.fields(klass))
    if len(known_names.symmetric_difference(effective_table.keys())) != 0:

        for k in effective_table:
            if k not in known_names:
                raise ValueError(f"`{table_name}.{k}` isn't a known parameter")
        for k in known_names:
            if k not in effective_table:
                raise ValueError(f"`{table_name}.{k}` is missing!")
    return klass(**effective_table)

def path_conf_from_file(file: os.PathLike | io.IOBase) -> PathConf:
    try:
        with open(file, "rb") as f:
            data = tomllib.load(f)
    except TypeError:
        data = tomllib.load(file)
    return _careful_init(data, "path", PathConf, {"simdata_prefix" : None})

def path_conf_from_cli(args: argparse.Namespace) -> PathConf:
    if args.conf is None:
        raise RuntimeError("--conf not provided")
    return path_conf_from_file(args.conf)

