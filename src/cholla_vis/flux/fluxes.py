"""
Define some functionality related to using the processed flux data.

These data products were created by logic in flux_process.py
"""

from dataclasses import dataclass
import os
from types import MappingProxyType

import numpy as np
import h5py
import unyt

from ..registry import _get_processed_data_dir

_MEASUREKIND_IDX_MAP = MappingProxyType(
    {'net' : 0, 'outflow' : 1, 'inflow' : 2}
)

def _careful_get(mapping, key, key_descr):
    # tries to load value associated with key from mapping, provides
    # a nice error message on failure
    try:
        return mapping[key]
    except KeyError:
        raise ValueError(
            f"{key!r} is an invalid {key_descr}. The only accepted values "
            f"are {list(mapping)}"
        ) from None

def _immutable_arr_copy(arr):
    """make a copy of ``arr`` and mark the array as immutable"""
    tmp = arr.copy()
    tmp.flags.writeable = False
    return tmp

@dataclass(frozen=True, kw_only=True)
class LevelBins:
    """encapsulates the levels at which we measured the fluxes.

    In general we track spherical radius or absolute z height"""
    descr: str
    units: str
    # the center of each bin
    all_centers: np.ndarray
    # the edges of each bin
    all_edges: np.ndarray
    # I don't remember what this is for
    used_indices: np.ndarray

    def __post_init__(self):
        for attr in ['all_centers', 'all_edges', 'used_indices']:
            object.__setattr__(
                self, attr, _immutable_arr_copy(getattr(self, attr))
            )

@dataclass(frozen=True, kw_only=True)
class FluxData:
    """Represents the calculated flux data."""
    
    # 1D ordered array of all the times at which we have flux measurements
    # -> this the union of all of the times that we made measurement
    t_Myr: np.ndarray

    # a (3, len(t_Myr)) array.
    # -> the index along axis 0 corresponds to the kind of measurement
    #    (see _MEASUREKIND_IDX_MAP)
    # -> It holds True for every time (in self.t_Myr) when we actually have
    #    measurements (when its False, I believe that flux_data will hold a NaN)
    mask: np.ndarray

    # the actual flux data. In more detail, this maps "density", "momentum" & "energy"
    # to flux measurements
    #
    #         ┌───────────── corresponds to the kind of measurement
    #         │  ┌───────────── matches len(t_Myr)
    #         │  │  ┌───────────── matches len(level_bins.all_centers)
    #         │  │  │  ┌───────────── corresponds to temperature cuts
    #         │  │  │  │  ┌───────────── corresponds to other_selection_bounds.shape[0]
    #         │  │  │  │  │
    # shape: [3, T, L, 3, S]
    flux_data: MappingProxyType[str, unyt.unyt_array]

    # same shape as `flux_data`. Specifies the number of cells that contribute to the
    # calculation of the flux in a given bin
    cell_count: np.ndarray

    # the independent variable: either the spherical radius or absolute z height
    level_bins: LevelBins

    # indicates whether the data was binned by openning angle or cylindrical radius
    other_selection_quan: str
    # a 2D array that actually denotes the bins of the "other selection quantity"
    other_selection_bounds: np.ndarray

    # maybe add T_bounds?

    def __post_init__(self):
        # this gets executed write after __init__ finishes
        # -> since FluxData is supposed to be an immutable class, we iterate through
        #    the attributes that hold arrays, make deepcopies of the arrays and mark
        #    the copies as immutable
        _normal_attrs = [
            't_Myr', 'cell_count', 'mask', 'other_selection_bounds'
        ]
        for attr in _normal_attrs:
            object.__setattr__(
                self, attr, _immutable_arr_copy(getattr(self, attr))
            )
        tmp = {k : _immutable_arr_copy(v) for k,v in self.flux_data.items()}
        object.__setattr__(self, 'flux_data', MappingProxyType(tmp))

    def _get_measurekind_slc(self, kind, packed_arr, only_masked = False):
        # helper method used to help us implement other methods
        leading_idx = _careful_get(_MEASUREKIND_IDX_MAP, kind, "measurekind")
        if only_masked:
            idx = (leading_idx, self.mask[leading_idx, :], ...)
        else:
            idx = (leading_idx, ...)
        return packed_arr[idx]

    def get_mask_for_kind(self, kind: str):
        # kind is one of 'net', 'outflow', 'inflow'
        return self._get_measurekind_slc(kind, self.mask, only_masked=False)

    def t_Myr_subset(self, kind: str) -> np.ndarray:
        # kind is one of 'net', 'outflow', 'inflow'
        return self.t_Myr[self.get_mask_for_kind(kind)]

    def get_cell_count(self, kind: str, only_masked: bool=False) -> np.ndarray:
        """Access cell_count for the ``kind`` flavor of fluxes

        Parameters
        ----------
        kind : {'net', 'outflow', 'inflow'}
            The kind of flux data to access
        only_masked : bool, optional
            Discard all data values corresponding to times where we don't
            have fluxes.
        Returns
        -------
        np.ndarray
            A 4D array with the same shape as the value returned by
            ``get_flux_data`` (with comparable args). The value at index
            ``idx`` specifying the number of cells that contributed to
            the calculation of the flux value at that index
        """
        return self._get_measurekind_slc(kind, self.cell_count, only_masked=False)

    def known_quan(self) -> list[str]:
        """List of known quantities with flux data"""
        # usually this is just ["density", "momentum", "energy"]
        return list(self.flux_data.keys())

    def get_flux_data(
            self, quan: str, kind: str, only_masked: bool = False
        ) -> unyt.unyt_array:
        """Access flux data of the ``quan`` quantity of the ``kind`` flavor

        Parameters
        ----------
        quan : str
            One of the quantities returned by known_quan (e.g. "density",
            "momentum", or "energy")
        kind : {'net', 'outflow', 'inflow'}
            The kind of flux data to access
        only_masked : bool, optional
            Discard all data values corresponding to times where we don't
            have fluxes.
        Returns
        -------
        unyt.unyt_array
            A 4D array of fluxes.
            - axis 0 corresponds to the time of measurements
            - axis 1 corresponds to the levels above the galaxy at which
              the measurements were made (e.g. abs_z or spherical radius)
            - axis 2 corresponds to the various temperature cuts
            - axis 3 corresponds to the "other selection quantity cuts."
              This is generally cylindrical radius or openning angle
        """
        # quantity is generally one of 'density', 'momentum', 'energy'
        # the output array has the relevent units
        f = _careful_get(self.flux_data, quan, "quantity")
        return self._get_measurekind_slc(kind, f, only_masked=only_masked)


def load_flux_data(
    sim_name: str,
    *,
    dirpath: os.PathLike | None = None,
    inner_dir: os.PathLike | None = None
) -> FluxData:
    """load in the precomputed flux data."""

    if dirpath is None:
        assert inner_dir is not None
        path = os.path.join(_get_processed_data_dir(), inner_dir, sim_name) + ".h5"
    else:
        assert inner_dir is None
        path = os.path.join(dirpath, sim_name) + ".h5"

    def _load_level_bins(h5_file, descr):
        """
        """

        bin_center_units = h5_file.attrs['level_all_bin_centers_unit']
        bin_edge_units = h5_file.attrs['level_all_bin_edges_unit']
        if bin_center_units != bin_edge_units:
            AssertionError(f"""\
Something went wrong. There is an inconsistency in the saved units:
-> units associated with the values at the bin centers:
     {bin_center_units!r}
-> units associated with the values at the bin edges:
     {bin_edge_units!r}""")

        return LevelBins(
            descr=descr,
            units=bin_center_units,
            all_centers=h5_file.attrs['level_all_bin_centers'],
            all_edges=h5_file.attrs['level_all_bin_edges'],
            used_indices=h5_file.attrs['level_bin_indices']
        )


    with h5py.File(path, 'r') as h5_file:
        # we make a bit of an educated guess about the structure of the data
        if 'bounds (rcyl/1.2kpc)^2' in h5_file.attrs:
            level_bins = _load_level_bins(h5_file, 'absz')
            other_selection_quan = 'bounds (rcyl/1.2kpc)^2'
            other_selection_bounds = h5_file.attrs[other_selection_quan]
        elif 'open_angle_deg' in h5_file.attrs:
            level_bins = _load_level_bins(h5_file, 'spherical_radius')
            other_selection_quan = 'open_angle_deg'
            other_selection_bounds = h5_file.attrs[other_selection_quan]
            if other_selection_bounds.ndim == 1:
                other_selection_bounds = np.stack(
                    [np.zeros_like(other_selection_bounds),
                     other_selection_bounds],
                    axis = 1
                )
        else:
            raise RuntimeError("The file has an unexpected state")

        assorted_arrays = {}
        flux_data = {}
        for key, dset in h5_file.items():
            arr = dset[...]
            if key in ['density', 'momentum', 'energy']:
                units = dset.attrs['units']
                flux_data[key] = unyt.unyt_array(arr, units=units)
            elif key in ["t_Myr", "mask", "cell_count"]:
                assert 'units' not in dset.attrs
                assorted_arrays[key] = arr
            else:
                raise RuntimeError(f"unknown key: {key}")

        out = FluxData(
            flux_data=MappingProxyType(flux_data),
            level_bins=level_bins,
            other_selection_quan=other_selection_quan,
            other_selection_bounds=other_selection_bounds,
            **assorted_arrays
        )
    return out

