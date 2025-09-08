"""
Plot the scale-height and the average density
"""

import numpy as np
import unyt
import yt
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib import colors
from cycler import cycler

import argparse
from collections import ChainMap
from enum import StrEnum
import os
from types import MappingProxyType

from cholla_vis.conf import build_conf_parser, path_conf_from_cli
from cholla_vis import profile_creator
from cholla_vis.flux import flux_process
from cholla_vis.registry import get_intermediate_data_registry

# none of the following commented out stuff is used
# def _broadcast_to_grid(axis_vals, axis_index, grid_shape):
#     reshaping_idx = tuple(
#         slice(None) if i == axis_index else np.newaxis
#         for i in range(len(grid_shape))
#     )
#     return np.broadcast_to(axis_vals[reshaping_idx], grid_shape)
# 
# def plot_assorted(
#     ax,
#     prof,
#     data_name,
#     xaxis_name,
#     binT_index,
#     other_spatial_bin_index = None,
#     data_units = None,
#     **kwargs
# ):
#     xfield, axis_index, otherax_index = flux_process._get_axis_props(prof, xaxis_name)
# 
#     xvals = _broadcast_to_grid(prof.data[xfield], axis_index, prof.data.shape)
#     yvals = prof.data[data_name]
# 
#     if data_units is not None:
#         yvals = yvals.to(data_units)
# 
#     assert 0 <= binT_index < 4
#     idx = [slice(None), slice(None), binT_index]
#     if other_spatial_bin_index is not None:
#         idx[otherax_index] = other_spatial_bin_index
#     idx = tuple(idx)
#     return ax.plot(xvals[idx], yvals[idx], **kwargs)


# I can't remember what the following was originally used for

# def _show_scale_height(
#     prof3D,
#     rebin_cylindrical_radius_bins = False,
#     downsample_factor=3,
#     merge_temperature_bins = False
# ):
#     tmp = _extract_data_from_prof(
#         prof3D, 
#         rebin_cylindrical_radius_bins,
#         downsample_factor=downsample_factor,
#         merge_temperature_bins=merge_temperature_bins,
#         kind = 'scale_height'
#     )
#     fig,ax = plt.subplots(1,1)
#     if merge_temperature_bins:
#         zest, rbins = tmp
#         ax.stairs(zest.to('kpc').v, rbins.to('kpc').v)
#     else:
#         zest, rbins, T_bins = tmp
#         for i, label in enumerate(profile_creator._standard_T_binnames()):
#             ax.stairs(zest[:,i].to('kpc').v, rbins.to('kpc').v,
#                       label = label)
#     ax.legend()
#     ax.set_yscale('log')
#     ax.set_ylabel(r'$\hat{z}_H$ (kpc)')
#     ax.set_xlabel(r'$r_{\rm cyl}$ (kpc)')
#     ax.set_xlim(0.0, ax.get_xlim()[1])
#     ax.axhline(0.005, ls = '--', label = 'resolution')
# 
# class AxisFieldProps:
#     def __init__(self, prof, name, label):
#         field, _, _ = flux_process._get_axis_props(prof, name)
#         self.name = name
#         self.field = field
#         self.label = label
#         vals = prof.data[field[0], field[1]].ndview
#         norm = colors.Normalize(vals.min(), vals.max())
#         cmap = matplotlib.colormaps['magma']
#         self.mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
#         self.pairs = [
#             (i, cmap(norm(val))) for i, val in enumerate(vals)
#         ]
# 
# def _define_known_axis_props(prof):
#     props = []
#     for field in [prof.x_field, prof.y_field]:
#         if field[1] == 'cylindrical_radius':
#             prop = AxisFieldProps(
#                 prof, field[1], r'$r_{\rm cyl}\ [{\rm kpc}]$'
#             )
#         elif field[1] == 'absz':
#             prop = AxisFieldProps(prof, field[1], r'$|z|\ [{\rm kpc}]$')
#         elif field[1] == 'spherical_radius':
#             prop = AxisFieldProps(
#                 prof, field[1], r'$r_{\rm sph}\ [{\rm kpc}]$'
#             )
#         else:
#             raise RuntimeError(field)
#         props.append(prop)
#     return props
        
def _dirty_rebin_slices(data_ndim, bin_ax, old_bins, new_bins):
    # a quick and dirty function for rebinning a quantity (this is imperfect)
    assert data_ndim > bin_ax
    old_locs = 0.5*(old_bins[:-1] + old_bins[1:])

    slc_template = [slice(None) for _ in range(data_ndim)]
    slice_pairs = []
    old_i = 0

    for new_i in range(new_bins.size - 1):
        left, right = new_bins[new_i], new_bins[new_i + 1]

        if old_i >= old_locs.size or old_locs[old_i] >= right:
            break
        elif old_locs[old_i] < left: 
            continue
        oldbinslc_start = old_i
        while ((old_i+1) < old_locs.size) and (old_locs[old_i+1] < right):
            old_i+=1
        olddata_slc = slc_template.copy()
        olddata_slc[bin_ax] = slice(oldbinslc_start, old_i+1)
        newdata_slc = slc_template.copy()
        newdata_slc[bin_ax] = new_i
        slice_pairs.append( (tuple(newdata_slc), tuple(olddata_slc)) )

        #print(old_locs[oldbinslc_start], old_locs[old_i], (left, right))
        old_i += 1

        
    return slice_pairs

def _dirty_rebin(data, bin_ax, old_bins, new_bins):
    new_shape = list(data.shape)
    new_shape[bin_ax] = new_bins.size - 1
    out = np.zeros(dtype = data.dtype, shape = new_shape)

    if isinstance(data, unyt.unyt_array):
        data_view = data.ndview
    else:
        data_view = data

    for outslc, inslc in _dirty_rebin_slices(data.ndim, bin_ax, old_bins, new_bins):
        out[outslc] = data_view[inslc].sum(axis=bin_ax)
    if isinstance(data, unyt.unyt_array):
        return out * data.uq
    return out


def _use_data_prefix(prof):
    return any(fkind == "data" for fkind, _ in prof.field_list)

class DataKind(StrEnum):
    """
    Specifies the kinds of data that can be derived from a scale-height phase dataset
    """
    SCALE_HEIGHT="scale_height"
    AVG_DENSITY="avg_density"
    VCIRC="vcirc"

def _extract_data_from_prof(
    prof3D,
    kind: DataKind,
    rebin_cylindrical_radius_bins:bool=False,
    downsample_factor:int=3,
    merge_temperature_bins:bool = False,
    stop_absz_bin_edge_index:int | None = None,
):
    """
    Extracts a derived quantity from a "scale-height" profile

    As I recall, the scale-heigh profile holds multiple quantities and are binned
    - along cylindrical radius
    - along absz
    - with respect to the temperature bins

    This function always performs an operation along the absz axis. It can
    optionally perform an operation along the temperature bins axis.

    Parameters
    ----------
    prof3D
        A single yt 3D profile
    kind
        The quantity to derive from the profile
    rebin_cylindrical_radius:
        When True, we downsample the cylindrical radius bins
    downsample_factor
        Specifies how to downsample the cylindrical radius bins
    merge_temperature_bins
        When False, we preserve the various temperature bins. When True, the
        temperature bins are all merged together (i.e. there is 1 bin of all
        temperatures)
    stop_absz_bin_edge_index
        When specified, all `indices >= stop_absz_bin_edge_index` are omitted
        from the calculation.
    """


    # this if-statement exists to handle the case where we imediately call this
    # function after creating the 3D profile or if we loaded a 3D profile from disk
    #
    # if you ask me, the fact that we need different code for each case is a poor
    # design decision in yt
    if _use_data_prefix(prof3D):
        findex, fgas = "data", "data"
        profdata = prof3D.data
        x_bins, y_bins, z_bins = (
            profdata['x_bins'], profdata['y_bins'], profdata['z_bins']
        )
    else:
        findex, fgas = "index", "gas"
        profdata = prof3D
        x_bins, y_bins, z_bins = prof3D.x_bins, prof3D.y_bins, prof3D.z_bins
    assert prof3D.x_field == (findex, 'cylindrical_radius')
    assert prof3D.z_field == (fgas, 'temperature')

    if stop_absz_bin_edge_index is None:
        idx = (slice(None), ...)
    else:
        # if it is 0, then we aren't selecting any bins
        assert stop_absz_bin_edge_index > 0
        idx = (slice(None), slice(0, stop_absz_bin_edge_index), ...)

    if kind == "scale_height":
        # this is an estimate
        field_names = [(fgas, "mass_zsq"), (fgas, "cell_mass")]
        def fn(data): return np.sqrt(data["mass_zsq"]/data["cell_mass"])
    elif kind == "avg_density":
        field_names = [(findex, "volume"), (fgas, "cell_mass")]
        def fn(data): return data["cell_mass"] / data["volume"]
    elif kind == "vcirc":
        # the first field is the sum of velocities
        # the second field is the number of cells in the sum
        field_names = [(fgas, "velocity_cylindrical_theta"),
                       ("data", "used")]
        def fn(data):
            # I guess this is implicitly volume-weighted
            vcirc = data["velocity_cylindrical_theta"]
            out = np.full(shape = vcirc.shape, dtype = 'f8',
                          fill_value = np.nan) * vcirc.uq
            counts = data["used"].ndview
            w = counts > 0
            out.ndview[w] = vcirc[w].ndview/counts[w]
            return out
    else:
        raise ValueError("unrecognized kind")

    # everything in field_name should be "extensive quantites"
    data = {
        field_name[1] : profdata[field_name][idx].sum(axis=1)
        for field_name in field_names
    }

    if rebin_cylindrical_radius_bins:
        ann_spec = profile_creator._standard_annuli_spec(use_code_length=True)
        standard_xbins = ann_spec.radii_thru_max_radius_squared(x_bins.max()**2)
        #if str(x_bins.units) == 'code_length':
        #    print('give units')
        #    standard_xbins = unyt.unyt_array(standard_xbins, x_bins.units)
        #else:
        standard_xbins = unyt.unyt_array(standard_xbins, 'kpc')[::downsample_factor]

        for key in data:
            data[key] = _dirty_rebin(data[key], 0, x_bins, standard_xbins)
        x_bins = standard_xbins

    if merge_temperature_bins:
        for key in data:
            data[key] = data[key].sum(axis=-1)
    out = fn(data)

    if merge_temperature_bins:
        return out, x_bins
    else:
        return out, x_bins, z_bins


def collect_data(sim_name, path_conf, downsample_factor=3, kind = 'scale_height',
                 merge_temperature_bins = False):
    """
    This function collects the specified kinds of data from inidvidual scale-height
    phase-profile datasets.

    """
    datasets = flux_process.get_data_names(
        sim_name, 'scale-height', path_conf=path_conf)
    radial_bins, T_bins = None, None
    times_Myr = []
    data = []
    for snap in datasets.keys():
        times_Myr.append(snap/10)
        prof = yt.load(datasets[snap], hint = "YTProfileDataset")
        pack = _extract_data_from_prof(
            prof, rebin_cylindrical_radius_bins = True,
            downsample_factor=downsample_factor,
            merge_temperature_bins = merge_temperature_bins,
            kind = kind
        )
        if merge_temperature_bins:
            tmp, radial_bins = pack
        else:
            tmp, radial_bins, T_bins = pack
        data.append(tmp)

    unit = data[0].units
    data2 = []
    for e in data:
        assert e.units == unit
        data2.append(e.ndview)
    
    stacked_data = unyt.unyt_array(np.stack(data, axis = -1), unit)
    times = unyt.unyt_array(times_Myr, 'Myr')
    return {kind: stacked_data,
            'radial_bins' : radial_bins,
            'T_bins': T_bins,
            'times': times}

def plot_scale_heights(ax_arr, ts_data, skip_xlabel=False,
                       skip_ylabel=False,
                       omit_linelabel = False,
                       **kwargs):
    radial_bins = ts_data['radial_bins']
    times_Myr = ts_data['times'].to('Myr').ndview
    zest = ts_data['scale_height'].to('kpc').ndview
    assert zest.shape[0]+1 == ts_data['radial_bins'].size
    if ts_data['T_bins'] is not None:
        assert zest.shape[1]+1 == ts_data['T_bins'].size
    assert zest.shape[-1] == ts_data['times'].size

    for j, ax in enumerate(ax_arr):
        col_label =(
            '$' + str(ts_data['radial_bins'][j].v) +
            r'\leq \frac{r_{\rm cyl}}{\rm kpc} <'
            + str(ts_data['radial_bins'][j+1].v) +
            '$'
        )

        if ts_data['T_bins'] is None:
            idx_label_pairs = [
                ( (j, slice(None)), 'all T')
            ]
        else:
            idx_label_pairs = (
                ( (j, i, slice(None)), label )
                for i, label in enumerate(profile_creator._standard_T_binnames())
            )
    
        for idx, label in idx_label_pairs:
            auto_kwargs = {} if omit_linelabel else {'label' : label}

            all_kwargs = ChainMap(kwargs, auto_kwargs)
            ax_arr[j].plot(times_Myr, zest[idx], **all_kwargs)

        ax_arr[j].axhline(0.005, ls = ':')

        if not skip_xlabel:
            ax_arr[j].set_xlabel('t [Myr]\n' + col_label)

    ax_arr[0].set_yscale('log')
    if not skip_ylabel:
        ax_arr[0].set_ylabel(r'$\hat{z}_H$ (kpc)')
    print(ts_data.keys())
    print(ts_data['scale_height'].shape)
    ax_arr[0]

def plot_all_scale_heights(scale_height_data, nradial_bins,
                           singleTbin_scale_height_data=None):
    sim_sets = [
        ('708cube_GasStaticG-1Einj',
         ['708cube_GasStaticG-1Einj_restart-TIcool',
          '708cube_GasStaticG-1Einj_restartDelay-TIcool'],
         r'<fiducial>'),
        ('708cube_GasStaticG-2Einj',
         ['708cube_GasStaticG-2Einj_restart-TIcool'],
         r'<$2E_{\rm inj}$>')
    ]

    ncol = nradial_bins

    fig, ax_arr = plt.subplots(
        len(sim_sets), ncol, figsize = (4*ncol, 4*len(sim_sets)),
        sharex=True, sharey=True, squeeze=False
    )
    num_Tbins = 4
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax in ax_arr.flatten():
        ax.set_prop_cycle(cycler(color=default_colors[:num_Tbins]))
    for i, (sim2phase, sim3phase_l, simlabel) in enumerate(sim_sets):
        if sim2phase is not None:
            plot_scale_heights(
                ax_arr[i], scale_height_data[sim2phase],
                skip_xlabel = True,
                omit_linelabel = True,
                ls = '--'
            )
            if singleTbin_scale_height_data is not None:
                plot_scale_heights(
                    ax_arr[i], singleTbin_scale_height_data[sim2phase],
                    skip_xlabel = True,
                    omit_linelabel = True,
                    ls = '--',
                    color = 'k'
                )

        for ls, sim3phase in zip(['-', '-.'], sim3phase_l):
            if sim3phase is not None:
                plot_scale_heights(
                    ax_arr[i], scale_height_data[sim3phase],
                    skip_xlabel = (i+1) != len(sim_sets),
                    omit_linelabel = (ls != '-'),
                    ls = ls
                )
                if singleTbin_scale_height_data is not None:
                    plot_scale_heights(
                        ax_arr[i], singleTbin_scale_height_data[sim3phase],
                        skip_xlabel = True,
                        omit_linelabel = True,
                        ls = ls,
                        color = 'k'
                    )
        ax_arr[i,0].set_ylabel(f'{simlabel}\n{ax_arr[i,0].get_ylabel()}')
    ax_arr[0,0].legend(loc='lower left', fontsize=8,framealpha=1.0)
    fig.tight_layout()
    return fig, ax_arr

def _plot_avg_density(ax, ts_data, skip_xlabel=False,
                      skip_ylabel=False,
                      omit_linelabel = False,
                      **kwargs):
    radial_bins = ts_data['radial_bins']
    times_Myr = ts_data['times'].to('Myr').ndview
    rho_avg = (ts_data['avg_density']/unyt.mh_cgs).to('cm**-3').ndview
    assert rho_avg.shape[0]+1 == ts_data['radial_bins'].size
    if ts_data['T_bins'] is not None:
        assert rho_avg.shape[1]+1 == ts_data['T_bins'].size
    assert rho_avg.shape[-1] == ts_data['times'].size


    if ts_data['T_bins'] is None:
        idx_label_pairs = [
            ( (slice(None),), 'all T')
        ]
    else:
        idx_label_pairs = (
            ( (0, i, slice(None)), label )
            for i, label in enumerate(profile_creator._standard_T_binnames())
        )
    
    for idx, label in idx_label_pairs:
        auto_kwargs = {} if omit_linelabel else {'label' : label}

        all_kwargs = ChainMap(kwargs, auto_kwargs)
        ax.plot(times_Myr, rho_avg[idx], **all_kwargs)

        if not skip_xlabel:
            ax.set_xlabel('t [Myr]\n')
    ax.set_yscale('log')
    if not skip_ylabel:
        ax.set_ylabel(r'$\rho/m_H\ [{\rm cm}^{-3}]$')


def plot_all_avg_densities(avg_density_data, nradial_bins,
                           singleTbin_data=None):
    sim_sets = [
        ('708cube_GasStaticG-1Einj',
         ['708cube_GasStaticG-1Einj_restart-TIcool',
          '708cube_GasStaticG-1Einj_restartDelay-TIcool'
         ],
         r'<fiducial>'),
        ('708cube_GasStaticG-2Einj',
         ['708cube_GasStaticG-2Einj_restart-TIcool'],
         r'<$2E_{\rm inj}$>')
    ]

    ncol = nradial_bins

    fig, ax_arr = plt.subplots(
        len(sim_sets), ncol, figsize = (4*ncol, 4*len(sim_sets)),
        sharex=True, sharey=True, squeeze=False
    )
    num_Tbins = 4
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax in ax_arr.flatten():
        ax.set_prop_cycle(cycler(color=default_colors[:num_Tbins]))
    for i, (sim2phase, sim3phase_l, simlabel) in enumerate(sim_sets):
        if sim2phase is not None:
            _plot_avg_density(
                ax_arr[i,0], avg_density_data[sim2phase],
                skip_xlabel = True,
                omit_linelabel = True,
                ls = '--'
            )
            if singleTbin_data is not None:
                _plot_avg_density(
                    ax_arr[i,0], singleTbin_data[sim2phase],
                    skip_xlabel = True,
                    omit_linelabel = True,
                    ls = '--',
                    color = 'k'
                )

        for ls, sim3phase in zip(['-', '-.'], sim3phase_l):
            if sim3phase is not None:
                _plot_avg_density(
                    ax_arr[i,0], avg_density_data[sim3phase],
                    skip_xlabel = (i+1) != len(sim_sets),
                    omit_linelabel = (ls != '-'),
                    ls = ls
                )
                if singleTbin_data is not None:
                    _plot_avg_density(
                        ax_arr[i,0], singleTbin_data[sim3phase],
                        skip_xlabel = True,
                        omit_linelabel = True,
                        ls = ls,
                        color = 'k'
                    )
        ax_arr[i,0].set_ylabel(f'{simlabel}\n{ax_arr[i,0].get_ylabel()}')
    ax_arr[1,0].legend(loc='lower left', fontsize=8,framealpha=0.8)
    fig.tight_layout()
    return fig, ax_arr

parser = argparse.ArgumentParser(
    description=__doc__, parents=[build_conf_parser()] 
)

if __name__ == '__main__':
    args = parser.parse_args()
    path_conf = path_conf_from_cli(args)

    _FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
    yt.set_log_level(40)
 
    for kind in [DataKind.SCALE_HEIGHT, DataKind.AVG_DENSITY]:
        #data_downsample3 = {}
        data_downsample12 = {}
        data_downsample12_allT = {}
 
        sim_names = get_intermediate_data_registry(path_conf=path_conf).keys()
        for sim_name in sim_names:
            print()
            print(sim_name)
            data_downsample12[sim_name] = collect_data(
                sim_name, path_conf=path_conf, kind = kind, downsample_factor=12
            )
            data_downsample12_allT[sim_name] = collect_data(
                sim_name, path_conf=path_conf, kind = kind, downsample_factor=12,
                merge_temperature_bins = True
            )
 
        if kind == DataKind.SCALE_HEIGHT:
            fig, ax_arr = plot_all_scale_heights(
                data_downsample12,
                nradial_bins=1,
                singleTbin_scale_height_data = data_downsample12_allT
            )
            #plt.show()
            plt.savefig(f'{_FIGDIR}/scale-heights-1bin.png')
            plt.close('all')
 
        else:
            plot_all_avg_densities(
                data_downsample12, nradial_bins=1,
                singleTbin_data=None
                #singleTbin_data=data_downsample12_allT
            )
            plt.savefig(f'{_FIGDIR}/avg_density-1bin.png')
            plt.close('all')
