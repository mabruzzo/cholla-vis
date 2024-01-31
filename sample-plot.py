import numpy as np
import unyt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import glob
import multiprocessing
import os
import os.path

from IPython.lib.pretty import pretty

from utils import (
    concat_particles, concat_slice, write_concat_particles, write_concat_slice,
    load_2D_h5
)

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

def _getGE(dset,suffix, u):
    try:
        return dset[f'GE_{suffix}'][:]
    except KeyError:
        mx, my, mz = dset[f'mx_{suffix}'][:], dset[f'my_{suffix}'][:], dset[f'mz_{suffix}'][:]
        KE = 0.5 * ((mx*mx) + (my*my) + (mz*mz)) / dset[f'd_{suffix}'][:]
        return dset[f'E_{suffix}'][:] - KE

def _getTemp(dset, suffix, unit_obj):
    u = unit_obj
    ge, den = _getGE(dset,suffix,u), dset[f'd_{suffix}'][:]
    n = den * u.DENSITY_UNIT/(u.mu*u.MP)
    return ge * u.PRESSURE_UNIT*(u.gamma - 1)/(n*u.KB)

def _getPhat(dset, suffix, unit_obj):
    u = unit_obj
    return _getGE(dset, suffix, u) * u.PRESSURE_UNIT*(u.gamma - 1)/u.KB

def _getNdens(dset, suffix, unit_obj):
    u = unit_obj
    return dset[f'd_{suffix}'][:] * u.DENSITY_UNIT/(u.mu*u.MP)


_slice_presets = {
    'temperature' : (_getTemp, 'temperature',
                     dict(
                        imshow_kwargs = {'cmap' : 'magma',# 'tab20c',
                                         'vmin' : 3.5, 'vmax' : 9,
                                         'alpha' : 0.95},
                        cbar_label = r"$\log_{10} T$ [K]")
                    ),
    'phat'        : (_getPhat, r'$p/k_B$',
                     dict(
                        imshow_kwargs = {'cmap' : 'viridis',
                                         'vmin' : -4, 'vmax' : 5.5,
                                         'alpha' : 0.95},
                        cbar_label = r"$\log_{10} (p / k_B)\ [{\rm K}\, {\rm cm}^{-3}]$")
                    ),
    'ndens'       : (_getNdens, 'number density',
                     dict(
                        imshow_kwargs = {'cmap' : 'viridis',
                                         # this is a complete guess
                                         'vmin' : -5, 'vmax' : 5,
                                         'alpha' : 0.95},
                        cbar_label = r"$\log_{10} n\ [{\rm cm}^{-3}]$")
                    )
}

def _getColDensity_proj(proj_dset, suffix, unit_obj):
    u = unit_obj
    return proj_dset[f'd_{suffix}'][:] * u.LENGTH_UNIT * u.DENSITY_UNIT/(u.mu*u.MP)

def _getAvgTemperature_proj(proj_dset, suffix, unit_obj):
    # T_{suffix} has units of Kelvin * LENGTH_UNIT * DENSITY_UNIT
    # d_{suffix} has units of LENGTH_UNIT * DENSITY_UNIT
    return proj_dset[f'T_{suffix}'][:] / proj_dset[f'd_{suffix}'][:]

_proj_presets = {
    'column_density' : (_getColDensity_proj, 'column_density',
                        dict(imshow_kwargs = {'cmap' : 'viridis',
                                              'vmin' : 20.0, 'vmax' : 24.25,
                                             },
                             cbar_label = r"$\log_{10} N\ [{\rm cm}^{-2}]$")
                        ),
    'avg_temperature' : (_getAvgTemperature_proj, 'avg_temperature',
                        dict(imshow_kwargs = {'cmap' : 'magma',
                                              'vmin' : 3.5, 'vmax' : 7
                                             },
                             cbar_label = r"$\log_{10} \langle T \rangle_{\rm mass-weighted} [{\rm K}]$")
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

class SliceParticleSelection:

    def __init__(self, p_props, orientation, max_age = None, max_sliceax_abspos = None):
        # identify existing particles
        if max_age is None:
            p_idx = slice(None)
            selection_count = 'all'
        else:
            p_idx = p_props['age'] < max_age
            selection_count = str(p_idx.sum())
        print(f"selecting {selection_count} particles out of {max_age.size} based on age")

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

        self.particle_imx = pos_arrays[imx][mask]
        self.particle_imy = pos_arrays[imy][mask]
        self.sizes = sizes[mask]



def _get_particle_props_for_slice(dset, orientation, max_slice_ax_absval = None):
    x,y,z = _particle_pos_xyz(dset)


def doLogSlicePlot2(data, slc_particle_selection, title, extent,
                    imshow_kwargs = {},
                    make_cbar = True, cbar_label = None,
                    orientation = 'xz'):
    # from Orlando!

    if orientation == 'xz':
        x_ax_label,y_ax_label = ["x (kpc)", "z (kpc)"]
    elif orientation == 'xy':
        x_ax_label,y_ax_label = ["x (kpc)", "y (kpc)"]
    elif orientation == 'yz':
        x_ax_label,y_ax_label = ["y (kpc)", "z (kpc)"]
    else:
        raise RuntimeError(f"unknown orientation: {orientation}")

    fig,ax = plt.subplots(1,1, figsize=(12, 10))
    ax.set_xlabel(x_ax_label)
    ax.set_ylabel(y_ax_label)

    ax.tick_params(axis='both', which='major')

    img=ax.imshow(np.log10(data.T),
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
    else:
        cax = None
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    return fig,ax,cax


def _rounded_float_to_str(v, decimals = 0):
    return repr(np.round(v, decimals))

def _make_2d_plot(hdr, slc_dset, p_props, u_obj,
                  kind = "slice",
                  preset_name = "temperature",
                  orientation = 'xz'):

    assert kind in ["slice", "proj"]
    cur_t = hdr['t'][0]

    pretty_t_str = f'{_rounded_float_to_str(cur_t/1000, 3)} Myr'

    if kind == "slice":
        presets = _slice_presets
    elif kind == "proj":
        presets = _proj_presets
    else:
        raise ValueError(f"unknown plot-kind: {kind!r}")
    if preset_name not in presets:
        raise ValueError(
            f"preset_name is unknown: {preset_name}. Known preset names "
            f"for {kind}-plots include {list(presets.keys())}")
    fn, quan_name, plot_kwargs = presets[preset_name]

    left_edge, domain_dims = hdr['bounds'], hdr['domain']

    (im_x_ind,im_y_ind), _ = _orient_to_imx_imy_zax(orientation)

    assert orientation in ['xz', 'yz','xy']
    if (p_props is None) or (kind != "slice"):
        slc_particle_selection = None
    else:
        max_sliceax_abspos = None if orientation == 'xy' else 0.4
        slc_particle_selection = SliceParticleSelection(
            p_props = p_props, orientation = orientation,
            max_age = cur_t,
            max_sliceax_abspos = max_sliceax_abspos)

    extent = (left_edge[im_x_ind], left_edge[im_x_ind] + domain_dims[im_x_ind],
              left_edge[im_y_ind], left_edge[im_y_ind] + domain_dims[im_y_ind])

    fig, ax, cax = doLogSlicePlot2(
        fn(slc_dset, orientation, u_obj),
        slc_particle_selection = slc_particle_selection,
        title = f'{quan_name} at {pretty_t_str}',
        extent = extent, orientation = orientation,
        **plot_kwargs)
    return fig

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
                    skip_particles = True):
 
    p_props = None
    if plot_proj:
        kind = "proj"
        if load_distributed_files:
            raise RuntimeError()
        else:
            path = f'{dnamein}/{n}_proj.h5'
            print(path)
            hdr, dset = load_2D_h5(path)
    else:
        kind = "slice"
        if load_distributed_files:
            hdr, dset = concat_slice(n, dnamein)
        else:
            path = f'{dnamein}/{n}_slice.h5'
            hdr, dset = load_2D_h5(path)
        if not skip_particles:
            if not load_distributed_files:
                _, p_props = concat_particles(n, dnamein)
            else:
                raise RuntimeError()
                

    u_obj = ChollaUnits()

    out_l = []
    for orient, pn in itr_orientation_preset(orientation, preset_name):
        fig = _make_2d_plot(hdr, dset, p_props, u_obj = ChollaUnits(),
                            preset_name = pn, orientation = orient,
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
              load_distributed_files = False):
    callback = partial(_saver, n = n, try_makedirs = try_makedirs,
                       outdir_prefix = outdir_prefix)

    print(f"Processing SNAP: {n}")

    make_slice_plot(run_dir, n, preset_name = preset_name,
                    orientation = orientation,
                    callback = callback, plot_proj = plot_proj,
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
               orientation = 'xz', load_distributed_files = False, pool = None):
    # try to make the output directories ahead of time:
    for orient, pn in itr_orientation_preset(orientation, preset_name):
        _saver(fig = None, n = None, orientation = orient, preset_name = pn,
               outdir_prefix = outdir_prefix, try_makedirs = True)

    func = _FuncWrapper(make_plot, preset_name = preset_name,
                        orientation = orientation, try_makedirs = False,
                        run_dir = run_dir, plot_proj = plot_proj,
                        outdir_prefix = outdir_prefix,
                        load_distributed_files = load_distributed_files)

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
