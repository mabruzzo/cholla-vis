import numpy as np
import unyt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import glob
import multiprocessing
import os
from typing import Any, Dict, NamedTuple

#from IPython.lib.pretty import pretty

from utils import (
    concat_particles, concat_slice, concat_proj, load_2D_h5
)

def _calc_vec_rot(x, y, vec_x, vec_y):
    theta = np.arctan2(y, x)
    return (-np.sin(theta) * vec_x +
            np.cos(theta) * vec_y)

class ChollaUnits:
    def __init__(self):
        self.LENGTH_UNIT = 3.08567758e21
        self.TIME_UNIT = 3.15569e10
        self.MASS_UNIT = 1.98855e33
        self.DENSITY_UNIT = self.MASS_UNIT/(self.LENGTH_UNIT*self.LENGTH_UNIT*self.LENGTH_UNIT)
        self.PRESSURE_UNIT = self.DENSITY_UNIT*(self.LENGTH_UNIT/self.TIME_UNIT)**2
        self.MP = 1.672622e-24
        self.gamma = 1.6666667
        self.KB = 1.380658e-16
        self.mu = 0.6

    def km_per_s_per_code_velocity(self):
        km_per_LENGTH_UNIT = self.LENGTH_UNIT / 1e5
        return km_per_LENGTH_UNIT/self.TIME_UNIT

def _getGE(dset,suffix, u, domain_dims):
    try:
        return dset[f'GE_{suffix}'][:]
    except KeyError:
        mx, my, mz = dset[f'mx_{suffix}'][:], dset[f'my_{suffix}'][:], dset[f'mz_{suffix}'][:]
        KE = 0.5 * ((mx*mx) + (my*my) + (mz*mz)) / dset[f'd_{suffix}'][:]
        return dset[f'E_{suffix}'][:] - KE

def _getTemp(dset, suffix, unit_obj, domain_dims):
    u = unit_obj
    ge, den = _getGE(dset,suffix,u, domain_dims), dset[f'd_{suffix}'][:]
    n = den * u.DENSITY_UNIT/(u.mu*u.MP)
    return ge * u.PRESSURE_UNIT*(u.gamma - 1)/(n*u.KB)

def _getPhat(dset, suffix, unit_obj, domain_dims):
    u = unit_obj
    return _getGE(dset, suffix, u, domain_dims) * u.PRESSURE_UNIT*(u.gamma - 1)/u.KB

def _getNdens(dset, suffix, unit_obj, domain_dims):
    u = unit_obj
    return dset[f'd_{suffix}'][:] * u.DENSITY_UNIT/(u.mu*u.MP)

def _get_xyz(dset, suffix, unit_obj, domain_dims, only_xy=False):
    cur_shape = dset[f'd_{suffix}'].shape
    domain_shape = np.array([dset['d_xz'].shape[0],
                             dset['d_xy'].shape[1],
                             dset['d_xz'].shape[1]])
    cell_width = domain_dims / domain_shape
    # assume left edge is -0.5 * domain_dims
    left_edge = -0.5 * domain_dims

    def _get_1d_pos(axis):
        return (
            (np.arange(domain_shape[axis]) + 0.5) * cell_width[axis]
            + left_edge[axis]
        )

    unbroadcasted = {
        # unclear if "else clause" is -0.5*cell_width[i] OR +0.5*cell_width[i]
        name : _get_1d_pos(i) if name in suffix else (0.5 * cell_width[i])
        for i,name in enumerate('xyz')
    }

    if suffix == 'xz':
        assert cur_shape[0] % 2 == 0
        x_vals = np.broadcast_to(unbroadcasted['x'][None].T, shape = cur_shape)
        y_vals = np.broadcast_to(unbroadcasted['y'], shape = cur_shape)
        z_vals = np.broadcast_to(unbroadcasted['z'][None], shape = cur_shape)
    elif suffix == 'yz':
        assert cur_shape[0] % 2 == 0
        x_vals = np.broadcast_to(unbroadcasted['x'], shape = cur_shape)
        y_vals = np.broadcast_to(unbroadcasted['y'][None].T, shape = cur_shape)
        z_vals = np.broadcast_to(unbroadcasted['z'][None], shape = cur_shape)
    else:
        x_vals = np.broadcast_to(unbroadcasted['x'][None].T, shape = cur_shape)
        y_vals = np.broadcast_to(unbroadcasted['y'][None], shape = cur_shape)
        z_vals = np.broadcast_to(unbroadcasted['z'], shape = cur_shape)
    if only_xy:
        return x_vals, y_vals
    return x_vals, y_vals, z_vals

def _get_vcomp_km_per_s(dset, axis, suffix, unit_obj):
    return (
        dset[f'm{axis}_{suffix}'] / dset[f'd_{suffix}']
    ) * unit_obj.km_per_s_per_code_velocity()


def _getVrot(dset, suffix, unit_obj, domain_dims):
    x_vals, y_vals = _get_xyz(dset, suffix, unit_obj, domain_dims, only_xy=True)

    vel = _calc_vec_rot(x_vals, y_vals,
                        dset[f'mx_{suffix}'],
                        dset[f'my_{suffix}']) / dset[f'd_{suffix}']
    return vel * unit_obj.km_per_s_per_code_velocity()

def _getRcylSupport(dset, suffix, unit_obj, domain_dims):
    vrot2 = np.square(_getVrot(dset, suffix, unit_obj, domain_dims))

    x_vals, y_vals, z_vals = _get_xyz(dset, suffix, unit_obj, domain_dims)    
    u = unit_obj
    pressure = _getGE(dset, suffix, u, domain_dims) * (u.gamma - 1)

    def deriv(axis, p, pos):
        Pderiv = np.empty_like(p)
        if axis == 0:
            Pderiv[1:-1,:] = (p[2:,:] - p[:-2,:])/(pos[2:,:] - pos[:-2,:])
            Pderiv[0,:] = (p[1,:] - p[0,:])/(pos[1,:] - pos[0,:])
            Pderiv[-1,:] = (p[-1,:] - p[-2,:])/(pos[-1,:] - pos[-2,:])
        else:
            Pderiv[:,1:-1] = (p[:,2:] - p[:,:-2])/(pos[:,2:] - pos[:,:-2])
            Pderiv[:,0] = (p[:,1] - p[:,0])/(pos[:,1] - pos[:,0])
            Pderiv[:,-1] = (p[:,-1] - p[:,-2])/(pos[:,-1] - pos[:,-2])
        return Pderiv

    if suffix == 'xz':
        dPdx = deriv(0, pressure, x_vals)
        dPdy = 0.0 # an approximation, but its okay
    elif suffix == 'yz':
        dPdx = 0.0 # an approximation, but its okay
        dPdy = deriv(0, pressure, y_vals)
    else:
        dPdx = deriv(0, pressure, x_vals)
        dPdy = deriv(1, pressure, y_vals)

    rcyl_times_dPdr = (x_vals * dPdx + y_vals * dPdy)

    rcyl_times_dPdr_div_rho = rcyl_times_dPdr/dset[f'd_{suffix}']

    km_per_LENGTH_UNIT = unit_obj.LENGTH_UNIT / 1e5
    pressure_contrib = (rcyl_times_dPdr_div_rho * (km_per_LENGTH_UNIT/unit_obj.TIME_UNIT)**2)
    return vrot2 + - pressure_contrib

def _getMomentumMag(dset, suffix, unit_obj, domain_dims):
    u = unit_obj
    domain_shape = np.array([dset['d_xz'].shape[0],
                             dset['d_xy'].shape[1],
                             dset['d_xz'].shape[1]])
    cell_width = domain_dims / domain_shape
    cell_volume = np.prod(cell_width)

    momentum_dens_mag = np.sqrt(
        np.square(dset[f'mx_{suffix}']) +
        np.square(dset[f'my_{suffix}']) +
        np.square(dset[f'mz_{suffix}'])
    )
    momentum_mag = momentum_dens_mag * cell_volume

    km_per_LENGTH_UNIT = u.LENGTH_UNIT / 1e5
    out = momentum_mag * (km_per_LENGTH_UNIT / unit_obj.TIME_UNIT)
    return out

'''
class FieldPlotSpec(NamedTuple):
    fn : callable
    full_name:
    latex_repr : str # don't include '$'
    default_units_repr : str # don't include '$'
    take_log : bool
    imshow_kwargs: Dict[str, Any]

def _cbar_label(plot_spec : FieldPlotSpec):
    prefix, latex_repr_suffix = '$', ''
    if plot_spec.take_log:
        prefix = r'$\log_{10} ('
        latex_repr_suffix = ')'

    return (f'{prefix} {plot_spec.latex_repr}{latex_repr_suffix} '
            f'[{plot.splecdefault_units_repr}]$')
'''

_slice_presets = {
    'temperature' : (_getTemp, 'temperature',
                     dict(
                        imshow_kwargs = {'cmap' : 'plasma',# 'tab20c',
                                         'vmin' : 1.0, # 3.5,
                                         'vmax' : 9,
                                         'alpha' : 0.95},
                        cbar_label = r"$\log_{10} T$ [K]",
                        take_log = True)
                    ),
    'phat'        : (_getPhat, r'$p/k_B$',
                     dict(
                        imshow_kwargs = {'cmap' : 'viridis',
                                         'vmin' : 2, 'vmax' : 5.5,
                                         'alpha' : 0.95},
                        cbar_label = r"$\log_{10} (p / k_B)\ [{\rm K}\, {\rm cm}^{-3}]$",
                        take_log = True)
                    ),
    'ndens'       : (_getNdens, 'number density',
                     dict(
                        imshow_kwargs = {'cmap' : 'plasma',
                                         # this is a complete guess
                                         'vmin' : -3, 'vmax' : 3,
                                         'alpha' : 0.95},
                        cbar_label = r"$\log_{10} n\ [{\rm cm}^{-3}]$",
                        take_log = True)
                    ),
    'vrot'       : (_getVrot, 'rotational velocity',
                     dict(
                        imshow_kwargs = {#'cmap' : 'plasma',
                                         'cmap' : 'coolwarm',
                                         # this is a complete guess
                                         'vmin' : -400, 'vmax' : 400,
                                         'alpha' : 0.95},
                        cbar_label = r"$ v_{\rm rot} [{\rm km} {\rm s^{-1}}]$",
                        take_log = False)
                    ),
    'rcyl_support' : (_getRcylSupport, 'rcyl support',
                     dict(
                        imshow_kwargs = {'cmap' : 'plasma',
                                         # this is a complete guess
                                         'vmin' : -1 * 100**2, 'vmax' : 300**2,
                                         'alpha' : 0.95},
                        cbar_label = (
                            r"$\left(v_{\rm rot}^2 + "
                            r"\frac{R}{\rho}\ \left|\frac{dP}{dR}\right|"
                            r"\right)\ "
                            r"[{\rm km}^2 {\rm s}^{-2}]$"),
                        take_log = False)
                    ),
    'momentum_mag' : (_getMomentumMag, 'momentum magnitude',
                      dict(
                          imshow_kwargs = {'cmap' : 'plasma',
                                         # this is a complete guess
                                         'vmin' : -1, 'vmax' : 5,
                                         'alpha' : 0.95},
                        cbar_label = r"$ |p| [{\rm M}_\odot\ {\rm km}\ {\rm s}^{-1}]$",
                        take_log = True)
                    ),
}

def _getColDensity_proj(proj_dset, suffix, unit_obj, domain_dims):
    u = unit_obj
    return proj_dset[f'd_{suffix}'][:] * u.LENGTH_UNIT * u.DENSITY_UNIT/(u.mu*u.MP)

def _getAvgTemperature_proj(proj_dset, suffix, unit_obj, domain_dims):
    # T_{suffix} has units of Kelvin * LENGTH_UNIT * DENSITY_UNIT
    # d_{suffix} has units of LENGTH_UNIT * DENSITY_UNIT
    return proj_dset[f'T_{suffix}'][:] / proj_dset[f'd_{suffix}'][:]

_proj_presets = {
    'column_density' : (_getColDensity_proj, 'column_density',
                        dict(imshow_kwargs = {'cmap' : 'viridis',
                                              'vmin' : 20.0, 'vmax' : 24.25,
                                             },
                             cbar_label = r"$\log_{10} N\ [{\rm cm}^{-2}]$",
                             take_log = True)
                        ),
    'avg_temperature' : (_getAvgTemperature_proj, 'avg_temperature',
                        dict(imshow_kwargs = {'cmap' : 'plasma',#'magma',
                                              'vmin' : 3.5, 'vmax' : 7
                                              #'vmin' : 0.5, 'vmax' : 7
                                             },
                             cbar_label = r"$\log_{10} \langle T \rangle_{\rm mass-weighted} [{\rm K}]$",
                             take_log = True)
                        ),
}


def _orient_to_imx_imy_zax(orientation, return_ind = True):
    if orientation == 'xz':
        im_x_ind, im_y_ind = 0,2 #x,z
        zax = 1
    elif orientation == 'xy':
        im_x_ind, im_y_ind = 0,1 #x,y
        zax = 2
    elif orientation == 'yz':
        im_x_ind, im_y_ind = 1,2 #y,z
        zax = 0
    else:
        raise RuntimeError("only known orientations are 'xz' & 'xy' & 'yz'")

    if return_ind:
        return (im_x_ind, im_y_ind), zax
    else:
        tmp = 'xyz'
        return (tmp[im_x_ind], tmp[im_y_ind]), tmp[zax]

def _particle_pos_xyz(dset, idx = None):
    if idx is None:
        idx = slice(None)
    if 'x' in dset.keys():
        return dset['x'][idx], dset['y'][idx], dset['z'][idx]
    return dset['pos_x'][idx], dset['pos_y'][idx], dset['pos_z'][idx]


class ParticleSelector:

    def __init__(self, age_based = True):
        self.age_based = age_based

    def __call__(self, p_props, orientation, snap_time = None, max_sliceax_abspos = None):
        if self.age_based and (snap_time is not None):
            p_idx = p_props['age'] < snap_time
            selection_count = str(p_idx.sum())
            print(f"selecting {selection_count} particles out of {p_props['age'].size} based on age")
        else:
            p_idx = slice(None)
            print(f"selecting all {p_props['age'].size} particles")

        # determine sizes of particles that we will plot
        norm_mass = p_props['mass'][p_idx]/np.sum(p_props['mass'][p_idx])
        sizes = ((norm_mass)*1e5) ** (1/2)

        # fetch particle positions
        pos_arrays = _particle_pos_xyz(p_props, p_idx)

        (imx, imy), slcax = _orient_to_imx_imy_zax(orientation)

        if max_sliceax_abspos is None:
            mask = slice(None)
        else:
            assert max_sliceax_abspos > 0
            mask = np.abs(pos_arrays[slcax]) < max_sliceax_abspos

        return SliceParticleSelection(pos_arrays[imx][mask], pos_arrays[imy][mask],
                                      sizes[mask])

class SliceParticleSelection:

    def __init__(self, particle_imx, particle_imy, sizes):
        self.particle_imx = particle_imx
        self.particle_imy = particle_imy
        self.sizes = sizes



def _get_particle_props_for_slice(dset, orientation, max_slice_ax_absval = None):
    x,y,z = _particle_pos_xyz(dset)

def _add_ax_labels(ax, orientation):
    if orientation == 'xz':
        x_ax_label,y_ax_label = ["x (kpc)", "z (kpc)"]
    elif orientation == 'xy':
        x_ax_label,y_ax_label = ["x (kpc)", "y (kpc)"]
    elif orientation == 'yz':
        x_ax_label,y_ax_label = ["y (kpc)", "z (kpc)"]
    else:
        raise RuntimeError(f"unknown orientation: {orientation}")

    ax.set_xlabel(x_ax_label)
    ax.set_ylabel(y_ax_label)

def doFullLogSlicePlot2(ax, data, slc_particle_selection, title, extent,
                        imshow_kwargs = {},
                        make_cbar = True, cbar_label = None,
                        orientation = 'xz', take_log = True,
                        cax = None):
    # from Orlando!
    _add_ax_labels(ax, orientation)

    ax.tick_params(axis='both', which='major')

    if take_log:
        vals = np.log10(data.T)
    else:
        vals = data.T

    img=ax.imshow(vals,
                  extent=extent,
                  origin = 'lower',
                  **imshow_kwargs)
    if slc_particle_selection is not None:
        ax.scatter(slc_particle_selection.particle_imx,
                   slc_particle_selection.particle_imy, marker='*', c='#32da13',
                   s=slc_particle_selection.sizes)
    if title is not None:
        ax.text(.05, .9, title, horizontalalignment='left',
                color="white", transform=ax.transAxes)
    if make_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img, cax=cax)
        if cbar_label is not None:
            plt.ylabel(cbar_label)
    elif cax is not None:
        raise RuntimeError("you can't set make_cbar=False and provide a cax")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    return cax




def _rounded_float_to_str(v, decimals = 0):
    return repr(np.round(float(v), decimals))

def _make_2d_plot(hdr, slc_dset, p_props, u_obj,
                  kind = "slice",
                  preset_name = "temperature",
                  orientation = 'xz',
                  particle_selector = None,
                  fig = None, ax = None, cax = None,
                  override_fn = None):

    assert kind in ["slice", "proj"]
    cur_t = hdr['t'][0]

    pretty_t_str = f'{_rounded_float_to_str(cur_t/1000, 3)} Myr'

    if kind == "slice":
        presets = _slice_presets
    elif kind == "proj":
        presets = _proj_presets
    else:
        raise ValueError(f"unknown plot-kind: {kind!r}")

    left_edge, domain_dims = hdr['bounds'], hdr['domain']

    (im_x_ind,im_y_ind), _ = _orient_to_imx_imy_zax(orientation)

    assert orientation in ['xz', 'yz','xy']

    if override_fn is None:

        if preset_name not in presets:
            raise ValueError(
                f"preset_name is unknown: {preset_name}. Known preset names "
                f"for {kind}-plots include {list(presets.keys())}")
        fn, quan_name, plot_kwargs = presets[preset_name]

        if (p_props is None) or (kind != "slice"):
            slc_particle_selection = None
        else:
            assert particle_selector is not None
            max_sliceax_abspos = None if orientation == 'xy' else 0.4
            slc_particle_selection = particle_selector(
                p_props = p_props, orientation = orientation,
                snap_time = cur_t,
                max_sliceax_abspos = max_sliceax_abspos)
    else:
        assert preset_name is None


    extent = (left_edge[im_x_ind], left_edge[im_x_ind] + domain_dims[im_x_ind],
              left_edge[im_y_ind], left_edge[im_y_ind] + domain_dims[im_y_ind])

    if fig is None:
        assert ax is None
        fig,ax = plt.subplots(1,1, figsize=(12, 10))
    elif ax is None:
        raise RuntimeError("fig and ax must both be None or neither can be None")

    if override_fn is None:
        doFullLogSlicePlot2(
            ax, fn(slc_dset, orientation, u_obj, domain_dims = domain_dims),
            slc_particle_selection = slc_particle_selection,
            title = f'{quan_name} at {pretty_t_str}',
            extent = extent, orientation = orientation,
            cax = cax,
            **plot_kwargs)
    else:
        override_fn(
            ax=ax, slc_dset=slc_dset, orientation=orientation, u_obj=u_obj,
            domain_dims=domain_dims, extent=extent
        )
    return fig

def _get_hdr_dset_kind(n, dnamein, plot_proj,
                       load_distributed_files = False):
    if plot_proj:
        kind = "proj"
        if load_distributed_files:
            hdr, dset = concat_proj(n,dnamein)
        else:
            path = f'{dnamein}/{n}_proj.h5'
            hdr, dset = load_2D_h5(path)
    else:
        kind = "slice"
        if load_distributed_files:
            hdr, dset = concat_slice(n, dnamein)
        else:
            path = f'{dnamein}/{n}_slice.h5'
            hdr, dset = load_2D_h5(path)
    return hdr, dset, kind

def make_slice_panel(fig,ax, dnamein, n, preset_name, orientation,
                     plot_proj = False,
                     load_distributed_files = False,
                     override_fn = None, cax = None):
    hdr, dset, kind = _get_hdr_dset_kind(
        n, dnamein, plot_proj, load_distributed_files
    )
    return _make_2d_plot(hdr, dset, p_props=None, u_obj = ChollaUnits(),
                         preset_name = preset_name, orientation = orientation,
                         particle_selector = None, kind = kind,
                         fig = fig, ax = ax, cax = cax,
                         override_fn = override_fn)
    

def itr_orientation_preset(orientation, preset_name):
    if isinstance(orientation, str):
        orient_l = [orientation]
    else:
        orient_l = orientation
    if isinstance(preset_name, str):
        preset_l = [preset_name]
    else:
        preset_l = preset_name

    for o in orient_l:
        for pn in preset_l:
            yield o, pn

def make_slice_plot(dnamein, n, preset_name, orientation,
                    callback = None, plot_proj = False,
                    load_distributed_files = False,
                    particle_selector = None):
 
    p_props = None
    hdr, dset, kind = _get_hdr_dset_kind(
        n, dnamein, plot_proj, load_distributed_files
    )
    if particle_selector is not None:
        if load_distributed_files:
            _, p_props = concat_particles(n, dnamein)
        else:
            raise RuntimeError()

    u_obj = ChollaUnits()

    out_l = []
    for orient, pn in itr_orientation_preset(orientation, preset_name):
        fig = _make_2d_plot(hdr, dset, p_props, u_obj = ChollaUnits(),
                            preset_name = pn, orientation = orient,
                            particle_selector = particle_selector,
                            kind = kind)

        if callback is None:
            out_l.append(fig)
        else:
            out_l.append(callback(orientation = orient, preset_name = pn, fig = fig))

    if isinstance(orientation, str) and isinstance(preset_name, str):
        assert len(out_l) == 1
        return out_l[0]
    else:
        return out_l

from functools import partial
import sys
from traceback import format_exc, format_tb


def _saver(fig, n, orientation, preset_name, outdir_prefix,
           try_makedirs = True):
    outdir = f'{outdir_prefix}/{orientation}/{preset_name}/'
    #print(outdir)
    if try_makedirs:
        os.makedirs(outdir, exist_ok = True)
    if (fig is not None) and (n is not None):
        fig.tight_layout()
        plt.savefig(f'{outdir}/{n:04d}.png', dpi = 300)
        plt.close(fig)
    return None

def make_plot(n, preset_name, run_dir, *, outdir_prefix,
              orientation = 'xz', try_makedirs = False, plot_proj = False,
              particle_selector = None,
              load_distributed_files = False):
    callback = partial(_saver, n = n, try_makedirs = try_makedirs,
                       outdir_prefix = outdir_prefix)

    #print(f"Processing SNAP: {n}")

    make_slice_plot(run_dir, n, preset_name = preset_name,
                    orientation = orientation,
                    callback = callback, plot_proj = plot_proj,
                    particle_selector = particle_selector,
                    load_distributed_files = load_distributed_files)
    return n

class _FuncWrapper:
    def __init__(self, fn, *args, **kwargs):
        self.partial_fn = partial(fn, *args, **kwargs)
    def __call__(self, *args, **kwargs):
        partial_fn = self.partial_fn
        try:
            return partial_fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            indent = '   '

            tmp = format_exc().replace('\n','\n'+indent) + '\n' + ('\n'+indent).join(format_tb(exc_traceback))
            print(
                f"error encountered during call with args = {args!r}, kwargs = "
                f"{kwargs!r}. The Error is:\n{indent}{tmp}\n",
                flush = True)

def make_plots(n_itr, preset_name, *, run_dir, plot_proj, outdir_prefix,
               orientation = 'xz', load_distributed_files = False, 
               particle_selector = None, pool = None):
    # try to make the output directories ahead of time:
    for orient, pn in itr_orientation_preset(orientation, preset_name):
        _saver(fig = None, n = None, orientation = orient, preset_name = pn,
               outdir_prefix = outdir_prefix, try_makedirs = True)

    func = _FuncWrapper(make_plot, preset_name = preset_name,
                        orientation = orientation, try_makedirs = False,
                        run_dir = run_dir, plot_proj = plot_proj,
                        outdir_prefix = outdir_prefix,
                        load_distributed_files = load_distributed_files,
                        particle_selector = particle_selector)

    if pool is None:
        my_map = map
    else:
        my_map = pool.imap_unordered

    count = 0
    for rslt in my_map(func, n_itr):
        count +=1
        print('done: ', rslt)

def main(args):

    #run_dir = ('/lustre/orion/ast181/scratch/mwabruzzo/disk-runs/'
    #           'first-attempt/2048/')
    #run_dir = ('../sample-data/2048/')

    load_distributed_files = False

    if args.kind == "proj":
        plot_proj, orientation = True, ['xy', 'xz']
        default_presets = ['column_density', 'avg_temperature']
    else:
        plot_proj, orientation = False, ['yz', 'xy', 'xz']
        default_presets = ['temperature', "ndens", 'phat']

    if args.save_dir is None:
        outdir_prefix = './images/'
    else:
        assert len(args.save_dir) > 0
        outdir_prefix = args.save_dir + '/'

    run_dir = args.load_dir

    snap_list = args.snaps

    if args.quan is None:
        preset_name = default_presets
    else:
        preset_name = args.quan

    def do_work(pool):
        try:
            make_plots(
                snap_list, run_dir = run_dir,
                preset_name = preset_name, orientation = orientation,
                plot_proj = plot_proj, pool = pool,
                outdir_prefix = outdir_prefix,
                load_distributed_files = load_distributed_files
            )
        except Exception as e:
            err_message = f'encountered problem'

            print(err_message)
            raise

    if args.ncores == 1:
        do_work(None)
    elif args.ncores > 1:
        with multiprocessing.Pool(processes=args.ncores) as pool:
            do_work(pool)
    else:
        raise RuntimeError('the --ncores arg must be a positive int')
# command line parsing
import argparse
import re

def integer_sequence(s):
    print(s)
    m = re.match(
        r"(?P<start>[-+]?\d+):(?P<stop>[-+]?\d+)(:(?P<step>[-+]?\d+))?",
        s)
    if m is not None:
        rslts = m.groupdict()
        step = int(rslts.get('step',1))
        if step == 0:
            raise ValueError(f"The range, {s!r}, has a stepsize of 0")
        seq = range(int(rslts['start']), int(rslts['stop']), step)
        if len(seq) == 0:
            raise ValueError(f"The range, {s!r}, has 0 values")
        return seq
    elif re.match(r"([-+]?\d+)(,[ ]*[-+]?\d+)+", s):
        seq = [int(elem) for elem in s.split(',')]
        return seq
    try:
        return [int(s)]
    except ValueError:
        raise ValueError(
            f"{s!r} is invalid. It should be a single int or a range"
        ) from None

parser = argparse.ArgumentParser(
    description = "Makes simple slice and projection plots from Cholla outputs"
)
parser.add_argument(
    '--snaps', type = integer_sequence, required=True,
    help = 'Which indices to plot. Can be a single number (e.g. 8) or '
           'a range specified with slice syntax (e.g. 2:9 or 5:3). ')
parser.add_argument(
    '--load-dir', type = str, default = './',
    help = 'Specifies directory to load data from')
parser.add_argument(
    '--save-dir', type = str, default = None,
    help = 'Specifies directory to save data in')
parser.add_argument(
    '-k', '--kind', choices=['slice', 'proj'], required=True,
    help = 'What kind of 2D dataset to plot')
parser.add_argument(
    '--quan', type = str, nargs = '+',
    help = 'The quantity to plot')
parser.add_argument(
    '--ncores', type = int, default = 1,
    help = 'Number of processes (using multiprocessing)')

if __name__ == '__main__':
    main(parser.parse_args())
