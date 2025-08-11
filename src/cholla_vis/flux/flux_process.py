import h5py
import numpy as np
import unyt
import yt

from dataclasses import dataclass
import os
from typing import NamedTuple

from .registry import get_intermediate_data_registry

def get_data_names(sim_name, data_name):
    path = get_intermediate_data_registry()[sim_name]
    l = []
    with os.scandir(os.path.join(path, data_name)) as it:
        for entry in filter(lambda e: e.is_file(), it):
            stem, ext = os.path.splitext(entry.name)
            assert ext == '.h5'
            l.append((int(stem), entry.path))
    l.sort()
    return dict(l)

class _AxisProp(NamedTuple):
    prof_field : tuple[str,str]
    axis_index : int
    otherax_index : int

    @property
    def prof_binfield(self):
        assert self.prof_field[1] in ['x', 'y', 'z']
        return (self.prof_field[0], f'{self.prof_field[1]}_bins')

def _get_axis_props(prof, axis_field_name):
    if isinstance(axis_field_name, str):
        def eq(prof_attr, axis_field_name):
            return prof_attr[1] == axis_field_name
    else:
        def eq(prof_attr, axis_field_name):
            return prof_attr == axis_field_name
    if eq(prof.x_field, axis_field_name):
        field, axis_index, otherax_index = ('data', 'x'), 0, 1
    elif eq(prof.y_field, axis_field_name):
        field, axis_index, otherax_index = ('data', 'y'), 1, 0
    else:
        raise RuntimeError(
            f"{axis_field_name} doesn't match either the profile's x_field, "
            f"{prof.x_field}, or y_field, {prof.y_field}"
        )
    return _AxisProp(field, axis_index, otherax_index)


def find_bin_index(val, bins):

    is_unyt = isinstance(val, unyt.unyt_array)
    assert is_unyt == isinstance(bins, unyt.unyt_array)
    if is_unyt:
        if val.units == bins.units:
            val = val.ndview
            bins = bins.ndview
        else:
            val = val.to(bins.units).ndview
            bins = bins.ndview

    nbins = len(bins)
    index = np.digitize(val, bins)
    # index of 0 corresponds to being on the left of the lowest bin_edge
    # index of nbins+1 corresponds to being on the right of the largest
    # bin_edge
    # -> effectively you could think of this as 1-based indexing of the bin
    assert np.all(index > 0)
    assert np.all(index <= nbins)

    # adjust to get a 0-based index of a bin
    return index - 1

@dataclass(frozen=True)
class _FluxDatasetDescr:
    # we have computed the flux at various "levels"
    # - you can think of this as a surface or level-set
    #   https://en.wikipedia.org/wiki/Level_set
    # - this isn't a perfect analog since we are actually binning the value
    level_field : tuple[str,str]

    # we commonly bin with respect to some alternative field
    alt_field: tuple[str, str]

    # maps 'density', 'momentum', and 'energy' to the appropriate field
    flux_field_map : dict[str, tuple[str,str]]

_FLUX_DATASET_DESCR_MAPPING = {
    'r_fluxes' : _FluxDatasetDescr(
        level_field = ('data', 'spherical_radius'),
        alt_field = ('data', 'symmetric_theta'),
        flux_field_map = {
            'density' : ('data', 'flux_rho_spherical_radius'),
            'momentum' : ('data', 'flux_momentum_spherical_radius'),
            'energy' : ('data', 'flux_e_spherical_radius'), 
        }
    ),
    'z_fluxes' : _FluxDatasetDescr(
        level_field = ('data', 'absz'),
        alt_field = ('data', 'cylindrical_radius'),
        flux_field_map = {
            'density' : ('data', 'flux_rho_outwardz'),
            'momentum' : ('data', 'flux_momentum_outwardz'),
            'energy' : ('data', 'flux_e_outwardz'), 
        }
    )
}

def _get_metadata(prof, target_level_vals, level_bins, dset_flux_props):
    level_field_axprop = _get_axis_props(prof, dset_flux_props.level_field)

    if target_level_vals is None:
        level_bin_indices = list(range(len(level_bins)-1))
    else:
        level_bin_indices = find_bin_index(
            target_level_vals, bins=level_bins
        )
    metadata = {
        'level_all_bin_edges' : level_bins,
        'level_all_bin_centers' : prof.data[level_field_axprop.prof_field],
        'level_bin_indices' : level_bin_indices
    }

    if level_field_axprop.axis_index == 0:
        def _build_idx(*, level_idx, T_idx, alt_field_slc):
            return (level_idx, alt_field_slc, T_idx)
    else:
        assert level_field_axprop.axis_index == 1
        def _build_idx(*, level_idx, T_idx, alt_field_slc):
            return (alt_field_slc, level_idx, T_idx)
    return level_bin_indices, metadata, _build_idx


def _gather_fluxes(
    sim_name,
    dataset_kind,
    snaps,
    choice = 'net',
    target_level_vals = None,
    alt_field_slc_l = None,
    T_idx_l = [slice(0,2), 2, 3]
):
    """
    the level field is used to define the isosurface (think of
    spherical radius or z position)

    target_level_vals: Optional
        The target level values
    """
    if choice == 'net':
        datasets = get_data_names(sim_name, dataset_kind)
    else:
        datasets = get_data_names(sim_name, dataset_kind + '_positive')
    if snaps is None:
        snaps = sorted(datasets.keys())

    #snaps = list(filter(_in_ranges, snaps))

    if alt_field_slc_l is None:
        alt_field_slc_l = [slice(None)]

    dset_flux_props = _FLUX_DATASET_DESCR_MAPPING[dataset_kind]
    flux_field_map = dset_flux_props.flux_field_map

    staging = {
        shorthand : [] for shorthand, field in flux_field_map.items()
    }
    staging['weight_sum'] = []


    t_Myr = []

    metadata = None

    num_snaps = len(snaps)
    print(num_snaps)

    full_shape = None
    

    for count, snap in enumerate(snaps):
        if count % 20 == 0:
            print(f" -> starting snap {count} of {num_snaps}")
        t_Myr.append(snap/10.0)
        prof = yt.load(datasets[snap], hint = "YTProfileDataset")

        level_field_axprop = _get_axis_props(prof, dset_flux_props.level_field)
        alt_field_axprop = _get_axis_props(prof, dset_flux_props.alt_field)

        level_bins = prof.data[level_field_axprop.prof_binfield]
        alt_bins = prof.data[alt_field_axprop.prof_binfield]

        if dset_flux_props.level_field == ('data','absz'):
            # sanity check!
            assert prof.x_field == ('data','absz')
            assert prof.y_field == ('data','cylindrical_radius')
            assert alt_bins[12].ndview == 1.2
        #else:
        #    raise RuntimeError(alt_bins, len(alt_bins))

        assert prof.data['data', 'z_bins'][2] == unyt.unyt_quantity(2e4, 'K')
        assert prof.data['data', 'z_bins'][3] == unyt.unyt_quantity(5e5, 'K')

        if metadata is None: # we are in our first pass
            level_bin_indices, metadata, _build_idx = _get_metadata(
                prof, target_level_vals, level_bins, dset_flux_props
            )

            accum_shape = (
                len(level_bin_indices), len(T_idx_l), len(alt_field_slc_l)
            )

        # the following could be substantially optimized if we used np.sum's
        # ability to pass in a tuple of indices to axis (we would just need to
        # understand the convention for the returned shape)
        # -> we could learn more about conventions from here
        #    https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduce.html
        all_weight_vals = prof.data['data', 'weight']
        accumulator = unyt.unyt_array(
            np.empty(shape=accum_shape, dtype='f8'),
            units=all_weight_vals.units
        )
        for i, level_idx in enumerate(level_bin_indices):
            for j, T_idx in enumerate(T_idx_l):
                for k, alt_field_slc in enumerate(alt_field_slc_l):
                    idx = _build_idx(
                        level_idx=level_idx,
                        T_idx=T_idx,
                        alt_field_slc=alt_field_slc
                    )
                    accumulator[i,j,k] = all_weight_vals[idx].sum()
        staging['weight_sum'].append(accumulator)

        for shorthand, field in flux_field_map.items():
            all_field_vals = prof.data[field]            
            accumulator = unyt.unyt_array(
                np.empty(shape=accum_shape, dtype='f8'),
                units=all_field_vals.units
            )
            for i, level_idx in enumerate(level_bin_indices):
                for j, T_idx in enumerate(T_idx_l):
                    for k, alt_field_slc in enumerate(alt_field_slc_l):
                        idx = _build_idx(
                            level_idx=level_idx,
                            T_idx=T_idx,
                            alt_field_slc=alt_field_slc
                        )
                        weights = all_weight_vals[idx]
                        field_vals = all_field_vals[idx]
                        weight_sum = weights.sum()
                        where = (weight_sum > 0.0)
                    
                        accumulator[i,j,k] = np.divide(
                            (field_vals * weights).sum(),
                            weight_sum,
                            where = where
                        )
                        accumulator[i,j,k][~where] = 0.0

            staging[shorthand].append(accumulator)

    # now we concatenate
    out = dict(
        t_Myr=np.array(t_Myr), **metadata
    )
    print('pre-stack')
    for field in staging:
        units = staging[field][0].units
        out[field] = unyt.unyt_array(
            np.stack(
                [tmp.to(units).ndview for tmp in staging[field]], axis=0
            ),
            units=units
        )
    print('post-stack')
    
    return out

def _convert_Myr_stamps_to_kyr(arr):
    return np.trunc(arr * 1000 +0.5).astype('i8')

def _compare_tkyr_array(ar1, ar2, *, inputs_in_Myr = False):
    # we assume that all tkyr entries are supposed to be integers

    if inputs_in_Myr:
        ar1 = _convert_Myr_stamps_to_kyr(ar1)
        ar2 = _convert_Myr_stamps_to_kyr(ar2)

    assert np.issubdtype(ar1.dtype, np.integer)
    assert np.issubdtype(ar2.dtype, np.integer)

    common_t_kyr, idx_ar1, idx_ar2 = np.intersect1d(
        ar1, ar2, assume_unique=True, return_indices=True
    )
    all_t_kyr = np.union1d(ar1, ar2)
    return common_t_kyr, idx_ar1, idx_ar2, all_t_kyr

def _get_cell_counts(aggregated_cell_counts, idx_net_outarray, net,
                     idx_outflow_outarray, outflow):
    # get cell-counts from "weight_sum"
    # deal with the "weight_sum:"
    # - net["weight_sum"] specifies the total number of cells in proximity of a
    #   surface
    # - outflow["weight_sum"] specifies the total number of cells, with
    #   outflowing fluid (in proximity of a surface)

    assert net['weight_sum'].units.is_dimensionless
    net_counts = net['weight_sum'].ndview
    assert outflow['weight_sum'].units.is_dimensionless
    outflow_counts = outflow['weight_sum'].ndview
    # sanity check!
    assert np.all(net_counts == np.trunc(net_counts))
    assert np.all(outflow_counts == np.trunc(outflow_counts))

    aggregated_cell_counts[0, idx_net_outarray, ...] = net_counts
    aggregated_cell_counts[1, idx_outflow_outarray, ...] = outflow_counts
    aggregated_cell_counts[2, ...] = (
        aggregated_cell_counts[0, ...] - aggregated_cell_counts[1, ...]
    )
    assert np.nanmin(aggregated_cell_counts) >= 0

def write_data(out_name, net, outflow):

    # common_t_kyr is the set of overlapping vals
    # all_t_kyr is the union of all t vals
    # -> idx_net_overlap and idx_outflow_overlap specify the respective sets of 
    #    indices that correspond to the overlapping times
    common_t_kyr, idx_net_overlap, idx_outflow_overlap, all_t_kyr = \
        _compare_tkyr_array(
            net['t_Myr'], outflow['t_Myr'], inputs_in_Myr = True
        )
    
    _kw = dict(assume_unique=True, return_indices=True)
    _, _, idx_net_outarray = np.intersect1d(
        _convert_Myr_stamps_to_kyr(net['t_Myr']), all_t_kyr, **_kw
    )
    _, _, idx_outflow_outarray = np.intersect1d(
        _convert_Myr_stamps_to_kyr(outflow['t_Myr']), all_t_kyr, **_kw
    )
    _, _, idx_inflow_outarray = np.intersect1d(
        common_t_kyr, all_t_kyr, assume_unique=True, return_indices=True
    )

    n_union = len(all_t_kyr)

    masks = np.ones( (3, n_union), dtype = np.bool_)

    case_labels = ['net', 'outflow', 'inflow']
    idx_l = [idx_net_outarray, idx_outflow_outarray, idx_inflow_outarray]

    # gather up the time stamps
    datasets = { 't_Myr' : all_t_kyr / 1000.0 }
    datasets['mask'] = np.zeros( (3, n_union), dtype = np.bool_)
    for (i, idx) in enumerate(idx_l):
        datasets['mask'][i,idx] = True

    # get cell-counts from "weight_sum"
    # deal with the "weight_sum:"
    shape = (3, n_union,) + net['weight_sum'].shape[1:]
    cell_count = np.full(shape = shape, fill_value = np.nan, dtype = 'f8')
    _get_cell_counts(cell_count, idx_net_outarray, net, idx_outflow_outarray, outflow)
    datasets['cell_count'] = cell_count

    # gather up the fluxes
    # -> the input data stores the average fluxes
    flux_keys = ['density', 'momentum', 'energy']
    flux_units = {k: str(net[k].units) for k in flux_keys}
    for key in flux_keys:
        u = flux_units[key]
        _shape = (3, n_union,) + net[key].shape[1:]
        if shape is None:
            shape = _shape
        else:
            assert shape == _shape
        agg = np.full(shape = shape, fill_value = np.nan, dtype = 'f8')

        net_avg = net[key].to(u).ndview
        agg[0, idx_net_outarray, ...] = net_avg
        net_total = cell_count[0, ...] * agg[0, ...]

        outflow_avg = outflow[key].to(u).ndview
        agg[1, idx_outflow_outarray, ...] = outflow_avg
        outflow_total = cell_count[1, ...] * agg[1, ...]

        inflow_total = (
            net_total[idx_inflow_outarray,...] - outflow_total[idx_inflow_outarray,...]
        )
        inflow_count = cell_count[2, idx_inflow_outarray, ...]
        w = inflow_count > 0
        inflow_avg = inflow_total.copy()
        inflow_avg[w] /= inflow_count[w]
        # originally, I wrote: inflow_avg[~w] = np.nan
        w_invalid = np.logical_or(np.isnan(inflow_count), inflow_count <= 0)
        inflow_avg[w_invalid] = np.nan
        inflow_avg[inflow_count == 0] = 0.0

        agg[2, idx_inflow_outarray, ...] = inflow_avg
        datasets[key] = agg

    with h5py.File(out_name,  'w') as f:
        for key in net:
            if (key in datasets) or key == 'weight_sum':
                continue
            elif key == 'label_props':
                attr_name, attr_val = net[key]
                f.attrs[attr_name] = attr_val
                continue
            else:
                assert np.all(net[key] == outflow[key])
                if isinstance(net[key], unyt.unyt_array):
                    f.attrs[key + '_unit'] = str(net[key].units)
                    f.attrs[key] = net[key].ndview
                else:
                    f.attrs[key] = net[key]
        for key in datasets:
            f[key] = datasets[key]
            if key in flux_units:
                f[key].attrs['units'] = flux_units[key]


def _collect(kind, choice = 'net', single_conf = True, sim_names = None):
    if kind == "z_fluxes":
        kwargs = dict(
            target_level_vals = None,
        )
        # specifiy the interval(s) of cylindrical radii to include
        if single_conf:
            alt_field_slc_l = []
            _label_l = []
        else:
            alt_field_slc_l = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)]
            _label_l = [(0, 1/4), (1/4, 1/2), (1/2, 3/4), (3/4, 1)]
        alt_field_slc_l.append(slice(0, 12)) # include the full disk!
        label_props = ('bounds (rcyl/1.2kpc)^2', _label_l + [(0, 1)])
    else:
        kwargs = dict()
        # specifiy the interval of symmetric_theta bins to include
        # - for reference, a cone with openning angle alpha includes
        #   all cells with symmetric_theta <= 0.5*alpha
        _AVAILABLE_SLC_L = [slice(0, i+1) for i in range(6)]
        _AVAILABLE_LABEL_VAL_L = [30,60,90,120,150,180]
        if single_conf:
            selection = slice(0,1) # only an openning angle of 30 degrees
        else:
            selection = slice(0,None)
        alt_field_slc_l = _AVAILABLE_SLC_L[selection]
        label_props = ('open_angle_deg', _AVAILABLE_LABEL_VAL_L[selection])

    yt.set_log_level(40)

    if sim_names is None:
        sim_names = [
            '708cube_GasStaticG-1Einj_restart-TIcool',
            '708cube_GasStaticG-1Einj',
            '708cube_GasStaticG-2Einj_restart-TIcool',
            '708cube_GasStaticG-2Einj'
        ]

    out = {}
    
    for sim_name in list(sim_names):
        print()
        print(sim_name, kind, choice)
        out[sim_name] = _gather_fluxes(
            sim_name, snaps = None,
            dataset_kind = kind,
            choice = choice,
            alt_field_slc_l = alt_field_slc_l,
            **kwargs
        )
        out[sim_name]['label_props'] = label_props
    return out

def _collect_and_save():
    prefix = '/ihome/eschneider/mwa25/cholla-bugfixing/galactic-center-analysis/'
    for sim_name in [
        '708cube_GasStaticG-1Einj_restart-TIcool',
        '708cube_GasStaticG-1Einj',
        '708cube_GasStaticG-2Einj_restart-TIcool',
        '708cube_GasStaticG-2Einj'
    ]:

        _kw = dict(single_conf = False, sim_names = [sim_name])
        
        rflux_datasets_net_cylrad = _collect("r_fluxes", **_kw)
        rflux_datasets_outflow_cylrad = _collect("r_fluxes", "positive", **_kw)

        print('aggregate and save r-fluxes')
        write_data(
            f'{prefix}/r_fluxes/{sim_name}.h5',
            net = rflux_datasets_net_cylrad[sim_name],
            outflow = rflux_datasets_outflow_cylrad[sim_name]
        )
        del rflux_datasets_net_cylrad
        del rflux_datasets_outflow_cylrad

        zflux_datasets_net_cylrad = _collect("z_fluxes", **_kw)
        zflux_datasets_outflow_cylrad = _collect("z_fluxes", "positive", **_kw)
        print('aggregate and save z-fluxes')
        write_data(
            f'{prefix}/z_fluxes/{sim_name}.h5',
            net = zflux_datasets_net_cylrad[sim_name],
            outflow = zflux_datasets_outflow_cylrad[sim_name]
        )
        del zflux_datasets_net_cylrad
        del zflux_datasets_outflow_cylrad

        print('==============================')

if __name__ == '__main__':
    _collect_and_save()
