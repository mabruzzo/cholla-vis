from collections.abc import Mapping
from functools import cache
import os
from types import MappingProxyType

from .conf import PathConf


@cache
def get_simdata_registry(
    name_prefix: str = '708cube_', *, path_conf: PathConf
) -> Mapping[str, os.PathLike]:
    """Returns a Mapping that maps simulation names to paths (to the directories that
    actually hold the simulation)"""

    def is_simdir(e):
        return e.is_dir() and e.name.startswith(name_prefix)

    if path_conf.simdata_prefix is None:
        return {}

    out = {}
    with os.scandir(path_conf.simdata_prefix) as it:
        for entry in filter(is_simdir, it):
            out[entry.name] = entry.path

    return MappingProxyType(out)


# TODO: we should rearrange probably the directory layout of the intermediate products
#       or the processed products so that they are more consistent!
#       They each have Pros and Cons:
#         - when you are developing a new data-product, it can be easier if the data
#           product names are the top level
#         - if you are adding new simulations, it can be easier of the simulation names
#           are the top level

@cache
def get_intermediate_data_registry(path_conf: PathConf) -> Mapping[str, os.PathLike]:
    """Returns a Mapping that maps simulation names to directory paths where
    intermediate data-products are stored

    Notes
    -----

    The following cartoon sketches roughly what the directories look like

        <prefix>/
        ├── <sim-name-0>/
        │    ├── <product-name-0>/   # <- name of an intermediate data product
        │    │    ├── <snap-num-A>.h5  # <- holds data for a snapshot (e.g. 0.h5)
        │    │    ├── ...              # <- holds data for another snapshot
        │    │    └── <snap-num-Z>.h5  # <- holds data for another snapshot
        │    ├── ...
        │    └── <product-name-m>/  # <- name of an intermediate data product
        │         ├── ...
        │         └── ...
        ├── ...
        └── <sim-name-N>/
            ├── ...
            └── ...
    """
    out = {}
    with os.scandir(path_conf.intermediate_data) as it:
        for entry in filter(lambda e: e.is_dir(), it):
            out[entry.name] = entry.path
    return MappingProxyType(out)


# we probably want to rethink this function
def _get_processed_data_dir(path_conf: PathConf) -> os.PathLike | None:
    """Returns the directory  that maps simulation names to directory paths where
    processed data-products are stored

    "Processed data-products" are generally more fully processed than
    "intermediate data products." In fact, the "Processed data-products"
    may have been computed from "intermediate data-products". The more
    significant distinction is the number of files per simulations.
    - In an "intermediate data-product," there is generally a separate
      file for each processed snapshot in a simulation
    - In a "processed data-product," there is generally a single file for
      the entire simulation.

    Notes
    -----
    The following cartoon sketches roughly what the directories look like

        <prefix>/
        ├── <product-name-0>/
        │    ├── <sim-name-a>.h5
        │    ├── ...
        │    └── <sim-name-z>.h5
        ├── ...
        │
        └── <product-name-n>/
            ├── ...
            └── ...
    """
    return path_conf.processed_data

