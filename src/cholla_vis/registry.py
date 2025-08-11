from collections.abc import Mapping
from functools import cache
import os
import platform
from types import MappingProxyType

# TODO: WE REALLY NEED TO WORK ON HOW THIS IS ORGANIZED

if 'oscar.local' == platform.node():
    _SIMDATA_PREFIX = None
    _INTERMED_PREFIX = '/Users/mabruzzo/Dropbox/research/mw-wind/data/analysis-data/'
    _PROCESSED_PREFIX = '/Users/mabruzzo/Dropbox/research/mw-wind/data/processed'
else:
    _SIMDATA_PREFIX = '/ix/eschneider/mabruzzo/hydro/galactic-center/'
    _INTERMED_PREFIX = '/ix/eschneider/mabruzzo/hydro/galactic-center/analysis-data/'
    _PROCESSED_PREFIX = None # <--- TODO: add me!

@cache
def get_simdata_registry(
    name_prefix: str = '708cube_', dir_path: os.PathLike | None = None
) -> Mapping[str, os.PathLike]:
    """Returns a Mapping that maps simulation names to paths (to the directories that
    actually hold the simulation)"""

    def is_simdir(e):
        return e.is_dir() and e.name.startswith(name_prefix)

    dir_path = _SIMDATA_PREFIX if dir_path is None else dir_path
    out = {}
    with os.scandir(dir_path) as it:
        for entry in filter(is_simdir, it):
            out[entry.name] = entry.path

    return MappingProxyType(_SIMDATA_REGISTRY)


# TODO: we should rearrange probably the directory layout of the intermediate products
#       or the processed products so that they are more consistent!
#       They each have Pros and Cons:
#         - when you are developing a new data-product, it can be easier if the data
#           product names are the top level
#         - if you are adding new simulations, it can be easier of the simulation names
#           are the top level

@cache
def get_intermediate_data_registry() -> Mapping[str, os.PathLike]:
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
    with os.scandir(_INTERMED_PREFIX) as it:
        for entry in filter(lambda e: e.is_dir(), it):
            out[entry.name] = entry.path
    return MappingProxyType(out)


# we probably want to rethink this function
def _get_processed_data_dir() -> os.PathLike | None:
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
    return _PROCESSED_PREFIX

