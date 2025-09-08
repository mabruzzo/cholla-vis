"""
Plot some flux data
"""


import h5py
import numpy as np
import pandas as pd
import unyt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import argparse
from dataclasses import dataclass
import os
from types import MappingProxyType
from typing import Any, NamedTuple

from cholla_vis.conf import build_conf_parser, path_conf_from_cli
from cholla_vis.registry import get_intermediate_data_registry
from cholla_vis.SNe_data import load_SNe_dataset
from cholla_vis.flux.fluxes import load_flux_data, FluxData
from cholla_vis.flux.ref_fluxes import bicone_surface_area, RefFluxCalculator

@dataclass
class _FundamentalQuanDescr:
    units: str
    # the following are used for formatting labels
    # -> each string will get enclosed by $ $
    # -> they should not include \frac
    var_latex: str
    unit_latex: str

_FUNDAMENTAL_QUAN_DESCR_MAP = {
    'density' : _FundamentalQuanDescr(
        units='Msun',
        var_latex = 'M',
        unit_latex = r'M_\odot'
    ),
    'momentum' : _FundamentalQuanDescr(
        units='(Msun * km/s)',
        var_latex = 'p',
        unit_latex = r'M_\odot\ {\rm km}\ {\rm s}^{-1}'
    ),
    'energy' : _FundamentalQuanDescr(
        units='erg',
        var_latex = 'E',
        unit_latex = r'{\rm erg}'
    )
}

def _derive_prop_map(fundamental_descr_map, for_fluxes):
    out = {}
    for key, d in fundamental_descr_map.items():
        if for_fluxes:
            units = f'({d.units})/kpc**2/yr'
            var_latex = rf'\mathcal{{F}}_{{{d.var_latex}}}'
            denom_u = r'{\rm kpc}^2\ {\rm yr}'
            unit_latex = rf'\frac{{ {d.unit_latex} }}{{{denom_u}}}'
        else:
            units = f'({d.units})/yr'
            var_latex = rf'\dot{{{d.var_latex}}}'
            unit_latex = rf'{d.unit_latex}\ {{\rm yr}}^{{-1}}'
        label = rf'${var_latex}\ \left[ {unit_latex} \right]$'
        out[key] = (units, label)
    return out

#derive_tderiv_props = functools.partial(
#    _derive_prop_map,
#    fundamental_descr_map=_FUNDAMENTAL_QUAN_DESCR_MAP,
#    for_fluxes=False
#)
#
#derive_flux_props = functools.partial(
#    _derive_prop_map,
#    fundamental_descr_map=_FUNDAMENTAL_QUAN_DESCR_MAP,
#    for_fluxes=True
#)


def _add_arbitrary_legend(
    ax, *, label_color_pairs=None, label_ls_pairs=None,
    line_color = None, line_ls = None,
    **legend_kwargs
):
    if label_color_pairs is not None:
        if label_ls_pairs is not None:
            raise ValueError(
                "Either label_color_pairs or label_ls_pairs can be specified "
                "(it is an error to provide both"
            )
        assert line_color is None
        line_ls = '-' if line_ls is None else line_ls
        custom_lines = [
            Line2D([0], [0], color=color, label=label)
            for label, color in label_color_pairs
        ]
    elif label_ls_pairs is not None:
        assert line_ls is None
        line_color = 'grey' if line_color is None else line_color
        custom_lines = [
            Line2D([0], [0], color=line_color, ls = ls, label=label)
            for label, ls in label_ls_pairs
        ]
    else:
        raise ValueError(
            "Either label_color_pairs or label_ls_pairs must be provided"
        )
    return ax.legend(handles = custom_lines, **legend_kwargs)

def _add_T_legend(ax, Tidx_color_pairs, line_ls=None, **legend_kwargs):
    labels = [
        r'$T < 2\times 10^4\, {\rm K}$',
        r'$2\times 10^4\, {\rm K} \leq T < 5\times 10^5\, {\rm K}$',
        r'$5\times 10^5\, {\rm K}\leq T$'
    ]

    return _add_arbitrary_legend(
        ax,
        label_color_pairs=[(labels[idx], c) for idx, c in Tidx_color_pairs],
        line_ls=line_ls,
        **legend_kwargs
    )

def _add_kind_legend(ax, choice_ls_map, line_color = None,
                     **legend_kwargs):

    return _add_arbitrary_legend(
        ax, label_ls_pairs=choice_ls_map.items(),
        line_color=line_color, **legend_kwargs
    )

def _find_index(val, arr):
    return np.argmax(np.isclose(val, arr))

def _averaged_flux_properties(
    target_t_Myr: float,
    quan: str,
    flux_data: FluxData | dict[str, unyt.unyt_array],
    duration_Myr: float,
    time_idx: Any = None,
    kind: str = None,
) -> unyt.unyt_array:
    """
    Computes the time-averaged flux properties

    Parameters
    ----------
    target_t_Myr
        Specifies the time (in Myr) when we want to know the flux
    quan: {"density", "momentum", "energy"}
        Specifies the quantity that we are considering.
    flux_data
        The flux data to use in the calculation. This can either be a
        ``FluxData`` instance (which directly stores measured data).
        Alternatively, this can be a ``dict`` (that holds somewhat
        manipulated data). In this case, we use ``flux_data[quan]``
    duration_Myr
        Specifies the duration of the average. A value of 0 uses the
        instantaneous value.
    time_idx
        An optional argument that can be used to only consider measurements
        from certain times.
    kind {"outflow", "inflow", "net"}, optional
        This must be specified when ``flux_data`` is a ``FluxData`` instance
        to indicate the kind of flux data to use. Otherwise, this must be
        ``None`` (the default).

    Returns
    -------
    unyt.unyt_array
        This holds the computed value. Recall that the input flux data is
        always an array with at least 1 dimension (corresponding to
        measurement time). The result is an array with one fewer dimensions. 
    """

    if isinstance(flux_data, FluxData):
        assert kind is not None
        t_Myr_vals = flux_data.t_Myr_subset(kind)
        flux_vals = flux_data.get_flux_data(quan, kind, only_masked=True)
    elif isinstance(flux_data, dict):
        t_Myr_vals = flux_data['t_Myr']
        flux_vals = flux_data[quan].T
    else:
        raise RuntimeError()

    assert duration_Myr >= 0
    # time runs along axis 0 of flux_vals
    if duration_Myr == 0:
        assert time_idx is None
        idx = _find_index(target_t_Myr, t_Myr_vals)
        assert np.isclose(t_Myr_vals[idx], target_t_Myr)
    else:
        lo = (target_t_Myr-0.5*duration_Myr)
        hi = (target_t_Myr+0.5*duration_Myr)
        #print(hi,lo)
        w_temp = np.logical_and(
            t_Myr_vals >= lo, t_Myr_vals < hi
        )
        if time_idx is None:
            idx = w_temp
        else:
            tmp = np.zeros(shape=t_Myr_vals, dtype=np.bool_)
            tmp[time_idx]=True
            idx = np.logical_and(tmp, w_temp)

    units = flux_vals.units
    return unyt.unyt_array(
        np.average(flux_vals.ndview[idx], axis=0),
        units = units
    )


def _coerce_units(a, units=None) -> unyt.unyt_array:
    """
    Coerce a to unyt.unyt_quantity

    Importantly, this also replaces code_length with kpc (which is accurate
    for cholla simulations)
    """
    view, myunits = a, units
    if units is None:
        assert isinstance(a, unyt.unyt_array)
        view, myunits = a.ndview, str(a.units)
    else:
        assert not isinstance(a, unyt.unyt_array)
    if myunits == 'code_length':
        myunits='kpc'
    if myunits != units:
        return unyt.unyt_array(view, myunits)
    return a

def _get_level_vals(flux_data) -> unyt.unyt_array:
    """
    Returns the level values of flux_data with appropriate units
    """
    return _coerce_units(
        flux_data.level_bins.all_centers,
        flux_data.level_bins.units
    )

# all of the following is currently unused:
#
# def _get_ref_fluxes(calculator, flux_data, flux_flavor):
#     level_vals = _get_level_vals(flux_data)
# 
#     if flux_flavor == 'r':
#         openning_angle = np.deg2rad(30.0)
#         out = calculator.r_flux(level_vals, openning_angle)
#     elif flux_flavor == 'z':
#         rcyl = np.ones_like(level_vals.ndview) * unyt.unyt_quantity(
#             1.2*np.sqrt(flux_data.other_selection_bounds.max()), 'kpc'
#         )
#         out = calculator.z_flux(rcyl)
#     else:
#         raise RuntimeError()
#     out['t_Myr'] = calculator.t_Myr
#     return out
# 
# def _plot_grid(target_t_Myr, ax_arr, calculator, flux_data, flux_flavor):
#     duration_Myr = 4.0
# 
#     ref_fluxes = _get_ref_fluxes(
#         _RefFlux_calculators['708cube_GasStaticG-1Einj_restart-TIcool'],
#         r_fluxes['708cube_GasStaticG-1Einj_restart-TIcool'],
#         flux_flavor
#     )
# 
#     Tidx_color_pairs = [(0, 'C0'),
#                         (1, 'C2'),
#                         (2, 'C3')]
# 
#     level_vals = _get_level_vals(flux_data)
# 
#     for i, (quan, conf) in enumerate(_FUNDAMENTAL_QUAN_DESCR_MAP.items()):
#         
#         ref_avg = _averaged_flux_properties(
#             target_t_Myr, quan, ref_fluxes, duration_Myr,
#             time_idx = None,
#             kind = None,
#             flux_kind = None,
#         )
#         ax_arr[i].set_ylabel(r'$\eta_{' + conf.var_latex + r'}$')
# 
#         _actual_avg = _averaged_flux_properties(
#             target_t_Myr, quan, flux_data, duration_Myr,
#             time_idx = None,
#             kind = 'net',
#             flux_kind = None,
#         )
#         if flux_flavor == 'r':
#             actual_avg = _actual_avg[...,0]
#         else:
#             actual_avg = _actual_avg[...,-1]
#         #print(actual_avg.shape)
# 
#         for Tidx, color in Tidx_color_pairs:
#             ratio = actual_avg[:, Tidx]/ref_avg
#             ax_arr[i].plot(level_vals, ratio, color = color)
# 
# def _test():
#     fig,ax_arr = plt.subplots(3,2, figsize = (8,12), sharex=True, sharey='row')
#     target_t_Myr = 110.0
#     _plot_grid(
#         target_t_Myr, ax_arr[:, 0],
#         _RefFlux_calculators['708cube_GasStaticG-1Einj_restart-TIcool'],
#         r_fluxes['708cube_GasStaticG-1Einj_restart-TIcool'],
#         'r'
#     )
# 
#     _plot_grid(
#         target_t_Myr, ax_arr[:, 1],
#         _RefFlux_calculators['708cube_GasStaticG-2Einj_restart-TIcool'],
#         r_fluxes['708cube_GasStaticG-2Einj_restart-TIcool'],
#         'r'
#     )

class StandardFluxLabelInfo(NamedTuple):
    # the order reflects the axis order
    time_label: str
    level_label: str
    Tbin_labels: list[str]
    selection_labels: list[str]


def _standard_flux_labels(flux_data: FluxData) -> StandardFluxLabelInfo:
    """
    returns a named tuple, ``out``, where ``out[i]`` holds the appropriate
    label for the ``i``th axis of the flux data

    This is very convenient for plotting
    """

    assert flux_data.other_selection_bounds.ndim == 2
    assert flux_data.other_selection_bounds.shape[1] == 2

    if flux_data.level_bins.descr == 'spherical_radius':
        assert (flux_data.other_selection_quan == 'open_angle_deg')
        level_label = r'$r_{\rm sph}$ [kpc]'
        open_angle_deg = flux_data.other_selection_bounds
        selection_labels = [
            rf'${lo}^{{\circ}} \leq \alpha_{{\rm open}} < {hi}^{{\circ}}$'
            for lo,hi in open_angle_deg
        ]
    elif flux_data.level_bins.descr == 'absz':
        assert flux_data.other_selection_quan == 'bounds (rcyl/1.2kpc)^2'
        level_label = r'$|z|$ [kpc]'
        is_standard = np.allclose(
            flux_data.other_selection_bounds,
            [[0.,0.25],[0.25,0.5],[0.5,0.75],[0.75,1.],[0.,1.]],
            rtol=1e-15, atol=0
        )
        if is_standard:
            ratio_str = r'\frac{r_{\rm cyl}}{1.2\ {\rm kpc}}'
            l = ['0', r'\sqrt{\frac{1}{4}}', r'\sqrt{\frac{1}{2}}',
                 r'\sqrt{\frac{3}{4}}', r'1']
            selection_labels = (
                [rf'${l[i]} \leq {ratio_str} < {l[i+1]}$' for i in range(4)] +
                [rf'${l[0]} \leq {ratio_str} < {l[4]}$']
            )
        else:
            raise RuntimeError()
    else:
        raise RuntimeError()
    return StandardFluxLabelInfo(
        time_label='t [Myr]',
        level_label=level_label,
        Tbin_labels=[
            r'$T < 2\times 10^4\, {\rm K}$',
            r'$2\times 10^4\, {\rm K} \leq T < 5\times 10^5\, {\rm K}$',
            r'$5\times 10^5\, {\rm K}\leq T$'
        ],
        selection_labels=selection_labels
    )


def _get_surface_area(flux_data: FluxData) -> unyt.unyt_array:
    """
    Returns an array of surface areas that match the shape of fluxes in `flux_data` if
    you are only considering a single time (or alternatively if you already averaged
    over time
    """
    level_vals = _get_level_vals(flux_data)
    assert flux_data.other_selection_bounds.ndim == 2
    len_other_selection_quan = flux_data.other_selection_bounds.shape[0]
    assert len_other_selection_quan > 0
    assert flux_data.other_selection_bounds.shape[1] == 2

    if flux_data.level_bins.descr == 'spherical_radius':
        level_axlen = level_vals.size # <- area
        assert (flux_data.other_selection_quan == 'open_angle_deg')
        def _areas_iter():
            r_spherical = level_vals
            open_angle_rad = np.deg2rad(flux_data.other_selection_bounds)
            for i in range(open_angle_rad.shape[0]):
                lo, hi = (
                    bicone_surface_area(r_spherical, open_angle_rad[i,j])
                    for j in [0,1]
                )
                cur_area = hi - lo
                yield cur_area
    elif flux_data.level_bins.descr == 'absz':
        level_axlen = 1 # <- we set this to 1 to support broadcasting
                        #    (since area is independent of absz)
        assert flux_data.other_selection_quan == 'bounds (rcyl/1.2kpc)^2'
        rcyl_sq = flux_data.other_selection_bounds * 1.44 * unyt.kpc**2
        def _areas_iter():
            for i in range(rcyl_sq.shape[0]):
                area = np.pi*(rcyl_sq[i,1] - rcyl_sq[i,0])
                assert area.ndim == 0 and area.size == 1
                yield area
    else:
        raise RuntimeError()

    outshape = (level_axlen,  1, len_other_selection_quan)
    #          rsph or absz,  T, other_component
    out_units = 'kpc**2'
    out = np.full(shape=outshape, fill_value=np.nan, dtype='f8')
    for i, area in enumerate(_areas_iter()):
        #print(i,area)
        out[:,0,i] = area.to(out_units).ndview
    return unyt.unyt_array(out, out_units)


def plot_flux_profile(
    ax: matplotlib.axes.Axes,
    flux_data: FluxData,
    quan: str,
    selection_idx: int,
    target_t_Myr: float,
    kind: str,
    duration_Myr=0,
    Tidx_color_pairs: list[tuple[int,str]] = [(0, 'C0'), (1, 'C2'), (2, 'C3')],
    ref_calculator: RefFluxCalculator | None =None,
    show_flux: bool|None = None,
    time_idx: Any =None,
    units: str|None = None,
    plot_kw: dict|None = None
):
    """
    Plots the specified flux data as a function of the appropriate "level values"

    For radial fluxes, fluxes are plotted against spherical radius. For z-fluxes,
    data is plotted against the displacement above the midplane.

    Parameters
    ----------
    ax
        A matplotlib Axes object to use for plotting the data.
    flux_data
        The data of interest is loaded from here
    quan: {"density", "momentum", "energy"}
        Specifies the quantity that we are considering.
    selection_idx
        The integer index of the selection quantity to consider. For radial
        flux data this corresponds to a choice in openning angle. For zfluxes
        this corresponds to a choice in cylindrical radius.
    target_t_Myr
        Specifies the time (in Myr) when we want to know the flux
    kind: {"outflow", "inflow", "net"}, optional
        This must be specified when ``flux_data`` is a ``FluxData`` instance
        to indicate the kind of flux data to use. Otherwise, this must be
        ``None`` (the default).
    duration_Myr
        Specifies the duration of the average. A value of 0 uses the
        instantaneous value.
    Tidx_color_pairs
        Specifies the colors to use for different temperature bins and how to
        color them.
    ref_calculator
        Optionally specifies a reference flux calculator. When provided, we
        plot loading factors.
    show_flux
        This should only be specified when ``ref_calculator`` is None. In this
        case, it accepts a bool to determine whether fluxes are plotted or
        global time derivatives are plotted.
    time_idx
        An optional argument that can be used to only consider measurements
        from certain times (this is relevant for averaging).
    plot_kw
        Can optionally specify a dictionary of extra kwargs to forward onto
        ``ax.plot``
    """
    plot_kw = {} if plot_kw is None else plot_kw

    conf = _FUNDAMENTAL_QUAN_DESCR_MAP[quan]

    if ref_calculator is not None:
        assert flux_data is not None
        assert show_flux is None
        _convert_to_time_deriv = True

        ref_time_deriv = ref_calculator.ref_time_deriv()
        ref_time_deriv['t_Myr'] = ref_calculator.t_Myr
        ref_avg = _averaged_flux_properties(
            target_t_Myr, quan, ref_time_deriv, duration_Myr,
            time_idx=None,kind=kind,
        )
        ylabel = rf'$\eta_{{{conf.var_latex}}}$'
        units = 'dimensionless'
    else:
        _convert_to_time_deriv = (show_flux is not None) and not show_flux
        units, ylabel = _derive_prop_map(
            _FUNDAMENTAL_QUAN_DESCR_MAP,
            for_fluxes=(not _convert_to_time_deriv)
        )[quan]
        ref_avg=None

    data = _averaged_flux_properties(
        target_t_Myr=target_t_Myr, quan=quan, flux_data=flux_data,
        duration_Myr=duration_Myr, time_idx=time_idx, kind=kind
    )

    if _convert_to_time_deriv:
        surface_area = _get_surface_area(flux_data)
        data = data * surface_area
        #print(surface_area)

    level_vals = _get_level_vals(flux_data)
    for j, (T_idx, color) in enumerate(Tidx_color_pairs):

        if ref_avg is None:
            yvals = data[:,j,selection_idx]
        else:
            yvals = data[:,j,selection_idx]/ref_avg
        #print(yvals)
        ax.plot(level_vals, yvals.to(units), color = color, **plot_kw)
    ax.set_ylabel(ylabel)


def generate_comparisons(shorthand_simname_pairs, fluxes, target_t_Myr,
                         duration_Myr, selection_idx, ref_calculators=None,
                         show_flux=None):
    """
    Generate plots comparing outflow measurements between simulations
    """
    kind_ls_pairs = [('net', '-'), ('outflow', '--'), ('inflow', '-.')]
    Tidx_color_pairs=[(0, 'C0'), (1, 'C2'), (2, 'C3')]

    assert len(shorthand_simname_pairs) == 2

    fig,ax_arr = plt.subplots(3,2, figsize=(6,8), sharex=True, sharey='row')

    labels = _standard_flux_labels(fluxes[shorthand_simname_pairs[0][1]])
    for ax in ax_arr[-1,:]:
        ax.set_xlabel(labels.level_label)

    for i, (shorthand, sim_name) in enumerate(shorthand_simname_pairs):
        ax_arr[0,i].set_title(
            shorthand + '\n' + labels.selection_labels[selection_idx]
        )
        for kind, ls in kind_ls_pairs: 
            for j, quan in enumerate(["density", "momentum", "energy"]):
                if ref_calculators is None:
                    ref_calculator = None
                else:
                    ref_calculator=ref_calculators[sim_name]
                plot_flux_profile(
                    ax_arr[j,i],
                    flux_data=fluxes[sim_name],
                    quan=quan,
                    kind=kind,
                    show_flux=show_flux,
                    selection_idx=selection_idx,
                    target_t_Myr=target_t_Myr,
                    duration_Myr=duration_Myr,
                    ref_calculator=ref_calculator,
                    plot_kw = {'ls' : ls},
                    Tidx_color_pairs=[(0, 'C0'), (1, 'C2'), (2, 'C3')],
                )

    _add_T_legend(ax_arr[1,0], Tidx_color_pairs, fontsize=8)
    _add_kind_legend(ax_arr[1,1], dict(kind_ls_pairs),
                     fontsize=8, line_color='k')

    fig.tight_layout()
    return fig, ax_arr

def mkdir_and_savefig(path,exist_ok=True,**kwargs):
    if '/' in path:
        os.makedirs(os.path.dirname(path), exist_ok=exist_ok)
    plt.savefig(path, **kwargs)

def _build_RefFlux_calculators(
    sne_datasets: dict[str, pd.DataFrame]
) -> dict[str, RefFluxCalculator]:
    """Construct a RefFluxCalculator for each SNe_dataset
    """

    def calc_SN_Einj(sim_name):
        # determine energy injected by a SN from the simulation name
        if '1Einj' in sim_name:
            return unyt.unyt_quantity(1e51, 'erg')
        elif '2Einj' in sim_name:
            return unyt.unyt_quantity(2e51, 'erg')
        else:
            raise RuntimeError()
    
    return {
        sim_name: RefFluxCalculator(
            sn_dataset=sn_dataset, use_lo=True, sn_e_inject=calc_SN_Einj(sim_name)
        ) for sim_name, sn_dataset in sne_datasets.items()
    }

parser = argparse.ArgumentParser(
    description=__doc__, parents=[build_conf_parser()]
)

if __name__ == '__main__':
    args = parser.parse_args()
    path_conf = path_conf_from_cli(args)

    _FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures", "fluxes")

    # load a list of simulation names
    sim_names = list(filter(
        lambda key: key!='708cube_GasStaticG-1Einj_restartDelay-TIcool',
        get_intermediate_data_registry(path_conf=path_conf).keys()
    ))

    # construct reference-flux calculators from the supernova rate datasets
    # stored from each simulation
    _SNe_datasets = {
        name : load_SNe_dataset(name, path_conf=path_conf) for name in sim_names
    }
    _RefFlux_calculators = _build_RefFlux_calculators(_SNe_datasets)

    # load the measured radial fluxes and the measured z fluxes
    r_fluxes = {
        sim_name : load_flux_data(sim_name, path_conf, inner_dir="r_fluxes")
        for sim_name in sim_names
    }
    z_fluxes = {
        sim_name : load_flux_data(sim_name, path_conf, inner_dir="z_fluxes")
        for sim_name in sim_names
    }

    # now we do a bunch of plotting of various combinations of quantities
    for my_dur, my_target_t_Myr in [
        (10, 115), (10, 125), (10, 135),
        (20, 115), (20, 125), (20, 135),
        (40, 125)
    ]:
        _my_kwarg = dict(
            shorthand_simname_pairs=[
                ('fid', '708cube_GasStaticG-1Einj_restart-TIcool'),
                (r'$2 E_{\rm inj}$', '708cube_GasStaticG-2Einj_restart-TIcool')
            ],
            target_t_Myr=my_target_t_Myr,
            duration_Myr=my_dur,
            ref_calculators=_RefFlux_calculators,
            #show_flux=False
        )
        _,ax_arr = generate_comparisons(
            fluxes=r_fluxes, selection_idx=0, **_my_kwarg)
        ax_arr[0,0].set_ylim(-0.1, 0.35)
        ax_arr[2,0].set_ylim(-0.001, 0.01)        
        mkdir_and_savefig(
            f'{_FIGDIR}/cmp_rflux/dur{my_dur}_{my_target_t_Myr}.png')
        plt.close('all')

        _,ax_arr = generate_comparisons(
            fluxes=z_fluxes, selection_idx=4, **_my_kwarg
        )
        ax_arr[0,0].set_ylim(-0.1, 0.35)
        ax_arr[2,0].set_ylim(-0.001, 0.01)        
        mkdir_and_savefig(
            f'{_FIGDIR}/cmp_zflux/dur{my_dur}_{my_target_t_Myr}.png')
        plt.close('all')

        _,ax_arr = generate_comparisons(
            fluxes=z_fluxes, selection_idx=0, **_my_kwarg
        )
        ax_arr[0,0].set_ylim(-0.1, 0.9)
        ax_arr[2,0].set_ylim(-0.001, 0.05)
        mkdir_and_savefig(
            f'{_FIGDIR}/cmp_zflux_inner/dur{my_dur}_{my_target_t_Myr}.png')

        plt.close('all')

    for my_dur, my_target_t_Myr in [
        (10, 115), (10, 125), (10, 135),
        (20, 115), (20, 125), (20, 135),
        (40, 125)
    ]:
        _my_kwarg = dict(
            shorthand_simname_pairs=[
                ('fid', '708cube_GasStaticG-1Einj_restart-TIcool'),
                (r'$2 E_{\rm inj}$', '708cube_GasStaticG-2Einj_restart-TIcool')
            ],
            target_t_Myr=my_target_t_Myr,
            duration_Myr=my_dur,
            #ref_calculators=_RefFlux_calculators,
            show_flux=False
        )
        _,ax_arr = generate_comparisons(
            fluxes=r_fluxes, selection_idx=0, **_my_kwarg)
        #ax_arr[0,0].set_ylim(-0.1, 0.35)
        #ax_arr[2,0].set_ylim(-0.001, 0.01)        
        mkdir_and_savefig(
            f'{_FIGDIR}/deriv_cmp_rflux/dur{my_dur}_{my_target_t_Myr}.png')
        plt.close('all')

        _,ax_arr = generate_comparisons(
            fluxes=z_fluxes, selection_idx=4, **_my_kwarg
        )
        #ax_arr[0,0].set_ylim(-0.1, 0.9)
        #ax_arr[2,0].set_ylim(-0.001, 0.05)
        mkdir_and_savefig(
            f'{_FIGDIR}/deriv_cmp_zflux/dur{my_dur}_{my_target_t_Myr}.png')
        plt.close('all')


    for my_dur, my_target_t_Myr in [
        (10, 115), (10, 125), (10, 135),
        (20, 115), (20, 125), (20, 135),
        (40, 125)
    ]:
        _my_kwarg = dict(
            shorthand_simname_pairs=[
                ('fid', '708cube_GasStaticG-1Einj_restart-TIcool'),
                (r'$2 E_{\rm inj}$', '708cube_GasStaticG-2Einj_restart-TIcool')
            ],
            target_t_Myr=my_target_t_Myr,
            duration_Myr=my_dur,
            #ref_calculators=_RefFlux_calculators,
            show_flux=True
        )
        _,ax_arr = generate_comparisons(fluxes=r_fluxes, selection_idx=0, **_my_kwarg)
        #ax_arr[0,0].set_ylim(-0.1, 0.35)
        #ax_arr[2,0].set_ylim(-0.001, 0.01)        
        mkdir_and_savefig(
            f'{_FIGDIR}/fluxvals_cmp_rflux/dur{my_dur}_{my_target_t_Myr}.png'
        )
        plt.close('all')

        _,ax_arr = generate_comparisons(
            fluxes=z_fluxes, selection_idx=4, **_my_kwarg
        )
        #ax_arr[0,0].set_ylim(-0.1, 0.9)
        #ax_arr[2,0].set_ylim(-0.001, 0.05)
        mkdir_and_savefig(
            f'{_FIGDIR}/fluxvals_cmp_zflux/dur{my_dur}_{my_target_t_Myr}.png')
        plt.close('all')

    #print(_get_surface_area(z_fluxes['708cube_GasStaticG-1Einj']))
    #print(_standard_flux_labels(z_fluxes['708cube_GasStaticG-1Einj']))


