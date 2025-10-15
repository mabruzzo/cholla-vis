import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from collections.abc import Callable, Iterable, Sequence
from functools import partial
import glob
import multiprocessing
import os
import sys
from traceback import format_exc, format_tb
from types import MappingProxyType
from typing import Any, NamedTuple
from .utils import concat_particles, concat_slice, concat_proj, load_2D_h5

try:
    from multiprocessing.pool import Pool as PoolType
except ImportError:
    type PoolType = Any


def _calc_vec_rot(x, y, vec_x, vec_y):
    theta = np.arctan2(y, x)
    return -np.sin(theta) * vec_x + np.cos(theta) * vec_y


class ChollaUnits:
    # holds hardcoded values assumed internally by Cholla
    def __init__(self):
        self.LENGTH_UNIT = 3.08567758e21
        self.TIME_UNIT = 3.15569e10
        self.MASS_UNIT = 1.98855e33
        self.DENSITY_UNIT = self.MASS_UNIT / (
            self.LENGTH_UNIT * self.LENGTH_UNIT * self.LENGTH_UNIT
        )
        self.PRESSURE_UNIT = (
            self.DENSITY_UNIT * (self.LENGTH_UNIT / self.TIME_UNIT) ** 2
        )
        self.MP = 1.672622e-24
        self.gamma = 1.6666667
        self.KB = 1.380658e-16
        self.mu = 0.6

    def km_per_s_per_code_velocity(self):
        km_per_LENGTH_UNIT = self.LENGTH_UNIT / 1e5
        return km_per_LENGTH_UNIT / self.TIME_UNIT


# These are some crude functions defined to compute derived fields that are relevant for
# plotting


def _getGE(dset, suffix, u, domain_dims):
    try:
        return dset[f"GE_{suffix}"][:]
    except KeyError:
        mx, my, mz = (
            dset[f"mx_{suffix}"][:],
            dset[f"my_{suffix}"][:],
            dset[f"mz_{suffix}"][:],
        )
        KE = 0.5 * ((mx * mx) + (my * my) + (mz * mz)) / dset[f"d_{suffix}"][:]
        return dset[f"E_{suffix}"][:] - KE


def _getTemp(dset, suffix, unit_obj, domain_dims):
    u = unit_obj
    ge, den = _getGE(dset, suffix, u, domain_dims), dset[f"d_{suffix}"][:]
    n = den * u.DENSITY_UNIT / (u.mu * u.MP)
    return ge * u.PRESSURE_UNIT * (u.gamma - 1) / (n * u.KB)


def _getPhat(dset, suffix, unit_obj, domain_dims):
    u = unit_obj
    return _getGE(dset, suffix, u, domain_dims) * u.PRESSURE_UNIT * (u.gamma - 1) / u.KB


def _getNdens(dset, suffix, unit_obj, domain_dims):
    u = unit_obj
    return dset[f"d_{suffix}"][:] * u.DENSITY_UNIT / (u.mu * u.MP)


def _get_xyz(dset, suffix, unit_obj, domain_dims, only_xy=False):
    cur_shape = dset[f"d_{suffix}"].shape
    domain_shape = np.array(
        [dset["d_xz"].shape[0], dset["d_xy"].shape[1], dset["d_xz"].shape[1]]
    )
    cell_width = domain_dims / domain_shape
    # assume left edge is -0.5 * domain_dims
    left_edge = -0.5 * domain_dims

    def _get_1d_pos(axis):
        return (np.arange(domain_shape[axis]) + 0.5) * cell_width[axis] + left_edge[
            axis
        ]

    unbroadcasted = {
        # unclear if "else clause" is -0.5*cell_width[i] OR +0.5*cell_width[i]
        name: _get_1d_pos(i) if name in suffix else (0.5 * cell_width[i])
        for i, name in enumerate("xyz")
    }

    if suffix == "xz":
        assert cur_shape[0] % 2 == 0
        x_vals = np.broadcast_to(unbroadcasted["x"][None].T, shape=cur_shape)
        y_vals = np.broadcast_to(unbroadcasted["y"], shape=cur_shape)
        z_vals = np.broadcast_to(unbroadcasted["z"][None], shape=cur_shape)
    elif suffix == "yz":
        assert cur_shape[0] % 2 == 0
        x_vals = np.broadcast_to(unbroadcasted["x"], shape=cur_shape)
        y_vals = np.broadcast_to(unbroadcasted["y"][None].T, shape=cur_shape)
        z_vals = np.broadcast_to(unbroadcasted["z"][None], shape=cur_shape)
    else:
        x_vals = np.broadcast_to(unbroadcasted["x"][None].T, shape=cur_shape)
        y_vals = np.broadcast_to(unbroadcasted["y"][None], shape=cur_shape)
        z_vals = np.broadcast_to(unbroadcasted["z"], shape=cur_shape)
    if only_xy:
        return x_vals, y_vals
    return x_vals, y_vals, z_vals


def _get_vcomp_km_per_s(dset, axis, suffix, unit_obj):
    return (
        dset[f"m{axis}_{suffix}"] / dset[f"d_{suffix}"]
    ) * unit_obj.km_per_s_per_code_velocity()


def _getVrot(dset, suffix, unit_obj, domain_dims):
    x_vals, y_vals = _get_xyz(dset, suffix, unit_obj, domain_dims, only_xy=True)

    vel = (
        _calc_vec_rot(x_vals, y_vals, dset[f"mx_{suffix}"], dset[f"my_{suffix}"])
        / dset[f"d_{suffix}"]
    )
    return vel * unit_obj.km_per_s_per_code_velocity()


def _getRcylSupport(dset, suffix, unit_obj, domain_dims):
    vrot2 = np.square(_getVrot(dset, suffix, unit_obj, domain_dims))

    x_vals, y_vals, z_vals = _get_xyz(dset, suffix, unit_obj, domain_dims)
    u = unit_obj
    pressure = _getGE(dset, suffix, u, domain_dims) * (u.gamma - 1)

    def deriv(axis, p, pos):
        Pderiv = np.empty_like(p)
        if axis == 0:
            Pderiv[1:-1, :] = (p[2:, :] - p[:-2, :]) / (pos[2:, :] - pos[:-2, :])
            Pderiv[0, :] = (p[1, :] - p[0, :]) / (pos[1, :] - pos[0, :])
            Pderiv[-1, :] = (p[-1, :] - p[-2, :]) / (pos[-1, :] - pos[-2, :])
        else:
            Pderiv[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (pos[:, 2:] - pos[:, :-2])
            Pderiv[:, 0] = (p[:, 1] - p[:, 0]) / (pos[:, 1] - pos[:, 0])
            Pderiv[:, -1] = (p[:, -1] - p[:, -2]) / (pos[:, -1] - pos[:, -2])
        return Pderiv

    if suffix == "xz":
        dPdx = deriv(0, pressure, x_vals)
        dPdy = 0.0  # an approximation, but its okay
    elif suffix == "yz":
        dPdx = 0.0  # an approximation, but its okay
        dPdy = deriv(0, pressure, y_vals)
    else:
        dPdx = deriv(0, pressure, x_vals)
        dPdy = deriv(1, pressure, y_vals)

    rcyl_times_dPdr = x_vals * dPdx + y_vals * dPdy

    rcyl_times_dPdr_div_rho = rcyl_times_dPdr / dset[f"d_{suffix}"]

    km_per_LENGTH_UNIT = unit_obj.LENGTH_UNIT / 1e5
    pressure_contrib = (
        rcyl_times_dPdr_div_rho * (km_per_LENGTH_UNIT / unit_obj.TIME_UNIT) ** 2
    )
    return vrot2 + -pressure_contrib


def _getMomentumMag(dset, suffix, unit_obj, domain_dims):
    u = unit_obj
    domain_shape = np.array(
        [dset["d_xz"].shape[0], dset["d_xy"].shape[1], dset["d_xz"].shape[1]]
    )
    cell_width = domain_dims / domain_shape
    cell_volume = np.prod(cell_width)

    momentum_dens_mag = np.sqrt(
        np.square(dset[f"mx_{suffix}"])
        + np.square(dset[f"my_{suffix}"])
        + np.square(dset[f"mz_{suffix}"])
    )
    momentum_mag = momentum_dens_mag * cell_volume

    km_per_LENGTH_UNIT = u.LENGTH_UNIT / 1e5
    out = momentum_mag * (km_per_LENGTH_UNIT / unit_obj.TIME_UNIT)
    return out


r"""
class FieldPlotSpec(NamedTuple):
    fn : callable
    full_name:
    latex_repr : str # don't include '$'
    default_units_repr : str # don't include '$'
    take_log : bool
    imshow_kwargs: dict[str, Any]

def _cbar_label(plot_spec : FieldPlotSpec):
    prefix, latex_repr_suffix = '$', ''
    if plot_spec.take_log:
        prefix = r'$\log_{10} ('
        latex_repr_suffix = ')'

    return (f'{prefix} {plot_spec.latex_repr}{latex_repr_suffix} '
            f'[{plot.splecdefault_units_repr}]$')
"""

_slice_presets = {
    "temperature": (
        _getTemp,
        "temperature",
        dict(
            imshow_kwargs={
                "cmap": "plasma",  # 'tab20c',
                "vmin": 1.0,  # 3.5,
                "vmax": 9,
                "alpha": 0.95,
            },
            cbar_label=r"$\log_{10} T$ [K]",
            take_log=True,
        ),
    ),
    "phat": (
        _getPhat,
        r"$p/k_B$",
        dict(
            imshow_kwargs={"cmap": "viridis", "vmin": 2, "vmax": 5.5, "alpha": 0.95},
            cbar_label=r"$\log_{10} (p / k_B)\ [{\rm K}\, {\rm cm}^{-3}]$",
            take_log=True,
        ),
    ),
    "ndens": (
        _getNdens,
        "number density",
        dict(
            imshow_kwargs={
                "cmap": "plasma",
                # this is a complete guess
                "vmin": -3,
                "vmax": 3,
                "alpha": 0.95,
            },
            cbar_label=r"$\log_{10} n\ [{\rm cm}^{-3}]$",
            take_log=True,
        ),
    ),
    "vrot": (
        _getVrot,
        "rotational velocity",
        dict(
            imshow_kwargs={  #'cmap' : 'plasma',
                "cmap": "coolwarm",
                # this is a complete guess
                "vmin": -400,
                "vmax": 400,
                "alpha": 0.95,
            },
            cbar_label=r"$ v_{\rm rot} [{\rm km} {\rm s^{-1}}]$",
            take_log=False,
        ),
    ),
    "rcyl_support": (
        _getRcylSupport,
        "rcyl support",
        dict(
            imshow_kwargs={
                "cmap": "plasma",
                # this is a complete guess
                "vmin": -1 * 100**2,
                "vmax": 300**2,
                "alpha": 0.95,
            },
            cbar_label=(
                r"$\left(v_{\rm rot}^2 + "
                r"\frac{R}{\rho}\ \left|\frac{dP}{dR}\right|"
                r"\right)\ "
                r"[{\rm km}^2 {\rm s}^{-2}]$"
            ),
            take_log=False,
        ),
    ),
    "momentum_mag": (
        _getMomentumMag,
        "momentum magnitude",
        dict(
            imshow_kwargs={
                "cmap": "plasma",
                # this is a complete guess
                "vmin": -1,
                "vmax": 5,
                "alpha": 0.95,
            },
            cbar_label=r"$ |p| [{\rm M}_\odot\ {\rm km}\ {\rm s}^{-1}]$",
            take_log=True,
        ),
    ),
}


def _getColDensity_proj(proj_dset, suffix, unit_obj, domain_dims):
    u = unit_obj
    return proj_dset[f"d_{suffix}"][:] * u.LENGTH_UNIT * u.DENSITY_UNIT / (u.mu * u.MP)


def _getAvgTemperature_proj(proj_dset, suffix, unit_obj, domain_dims):
    # T_{suffix} has units of Kelvin * LENGTH_UNIT * DENSITY_UNIT
    # d_{suffix} has units of LENGTH_UNIT * DENSITY_UNIT
    return proj_dset[f"T_{suffix}"][:] / proj_dset[f"d_{suffix}"][:]


_proj_presets = {
    "column_density": (
        _getColDensity_proj,
        "column_density",
        dict(
            imshow_kwargs={
                "cmap": "viridis",
                "vmin": 20.0,
                "vmax": 24.25,
            },
            cbar_label=r"$\log_{10} N\ [{\rm cm}^{-2}]$",
            take_log=True,
        ),
    ),
    "avg_temperature": (
        _getAvgTemperature_proj,
        "avg_temperature",
        dict(
            imshow_kwargs={
                "cmap": "plasma",  #'magma',
                "vmin": 3.5,
                "vmax": 7,
                #'vmin' : 0.5, 'vmax' : 7
            },
            cbar_label=r"$\log_{10} \langle T \rangle_{\rm mass-weighted} [{\rm K}]$",
            take_log=True,
        ),
    ),
}


def _get_known_presets(kind: str) -> MappingProxyType[str, Any]:
    if kind == "slice":
        return MappingProxyType(_slice_presets)
    elif kind == "proj":
        return MappingProxyType(_proj_presets)
    else:
        raise ValueError(f"unknown plot-kind: {kind!r}")


def _orient_to_imx_imy_zax(orientation, return_ind=True):
    if orientation == "xz":
        im_x_ind, im_y_ind = 0, 2  # x,z
        zax = 1
    elif orientation == "xy":
        im_x_ind, im_y_ind = 0, 1  # x,y
        zax = 2
    elif orientation == "yz":
        im_x_ind, im_y_ind = 1, 2  # y,z
        zax = 0
    else:
        raise RuntimeError("only known orientations are 'xz' & 'xy' & 'yz'")

    if return_ind:
        return (im_x_ind, im_y_ind), zax
    else:
        tmp = "xyz"
        return (tmp[im_x_ind], tmp[im_y_ind]), tmp[zax]


def _particle_pos_xyz(dset, idx=None):
    if idx is None:
        idx = slice(None)
    if "x" in dset.keys():
        return dset["x"][idx], dset["y"][idx], dset["z"][idx]
    return dset["pos_x"][idx], dset["pos_y"][idx], dset["pos_z"][idx]


class SliceParticleSelection(NamedTuple):
    particle_imx: Any  # array of positions along the image's x-axis
    particle_imy: Any  # array of positions along the image's y-axis
    sizes: Any  # array of values corresponding to marker sizes for
    # each particle


type ParticleSelectCallback = Callable[
    [dict[str, np.ndarray], str, float | None, int | None], SliceParticleSelection
]


class ParticleSelector:
    """
    A callback that selects relevant particles for generating a figure
    """

    def __init__(self, age_based: bool = True):
        self.age_based = age_based

    def __call__(
        self,
        p_props: dict[str, np.ndarray],
        orientation: str,
        snap_time: float | None = None,
        max_sliceax_abspos: int | None = None,
    ) -> SliceParticleSelection:
        if self.age_based and (snap_time is not None):
            p_idx = p_props["age"] < snap_time
            selection_count = str(p_idx.sum())
            print(
                f"selecting {selection_count} particles out of {p_props['age'].size} based on age"
            )
        else:
            p_idx = slice(None)
            print(f"selecting all {p_props['age'].size} particles")

        # determine sizes of particles that we will plot
        norm_mass = p_props["mass"][p_idx] / np.sum(p_props["mass"][p_idx])
        sizes = ((norm_mass) * 1e5) ** (1 / 2)

        # fetch particle positions
        pos_arrays = _particle_pos_xyz(p_props, p_idx)

        (imx, imy), slcax = _orient_to_imx_imy_zax(orientation)

        if max_sliceax_abspos is None:
            mask = slice(None)
        else:
            assert max_sliceax_abspos > 0
            mask = np.abs(pos_arrays[slcax]) < max_sliceax_abspos

        return SliceParticleSelection(
            pos_arrays[imx][mask], pos_arrays[imy][mask], sizes[mask]
        )


def _add_ax_labels(ax: plt.Axes, orientation: str):
    if orientation == "xz":
        x_ax_label, y_ax_label = ["x (kpc)", "z (kpc)"]
    elif orientation == "xy":
        x_ax_label, y_ax_label = ["x (kpc)", "y (kpc)"]
    elif orientation == "yz":
        x_ax_label, y_ax_label = ["y (kpc)", "z (kpc)"]
    else:
        raise RuntimeError(f"unknown orientation: {orientation}")

    ax.set_xlabel(x_ax_label)
    ax.set_ylabel(y_ax_label)


def doFullLogSlicePlot2(
    ax: plt.Axes,
    data: np.ndarray,
    slc_particle_selection: Any,
    title: str | None,
    extent: list[float],
    imshow_kwargs: dict[Any] = {},
    make_cbar: bool = True,
    cbar_label: str | None = None,
    orientation: str = "xz",
    take_log: bool = True,
    cax: plt.Axes | None = None,
) -> plt.Axes:
    # from Orlando!
    _add_ax_labels(ax, orientation)

    ax.tick_params(axis="both", which="major")

    if take_log:
        vals = np.log10(data.T)
    else:
        vals = data.T

    img = ax.imshow(vals, extent=extent, origin="lower", **imshow_kwargs)
    if slc_particle_selection is not None:
        ax.scatter(
            slc_particle_selection.particle_imx,
            slc_particle_selection.particle_imy,
            marker="*",
            c="#32da13",
            s=slc_particle_selection.sizes,
        )
    if title is not None:
        ax.text(
            0.05,
            0.9,
            title,
            horizontalalignment="left",
            color="white",
            transform=ax.transAxes,
        )
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


def _rounded_float_to_str(v, decimals=0):
    return repr(np.round(float(v), decimals))


def _make_2d_plot(
    hdr,
    slc_dset,
    p_props,
    u_obj,
    kind="slice",
    preset_name="temperature",
    orientation="xz",
    particle_selector=None,
    fig=None,
    ax=None,
    cax=None,
    override_fn=None,
):
    assert kind in ["slice", "proj"]
    cur_t = hdr["t"][0]

    pretty_t_str = f"{_rounded_float_to_str(cur_t / 1000, 3)} Myr"

    presets = _get_known_presets(kind=kind)

    left_edge, domain_dims = hdr["bounds"], hdr["domain"]

    (im_x_ind, im_y_ind), _ = _orient_to_imx_imy_zax(orientation)

    assert orientation in ["xz", "yz", "xy"]

    if override_fn is None:
        if preset_name not in presets:
            raise ValueError(
                f"preset_name is unknown: {preset_name}. Known preset names "
                f"for {kind}-plots include {list(presets.keys())}"
            )
        fn, quan_name, plot_kwargs = presets[preset_name]

        if (p_props is None) or (kind != "slice"):
            slc_particle_selection = None
        else:
            assert particle_selector is not None
            max_sliceax_abspos = None if orientation == "xy" else 0.4
            slc_particle_selection = particle_selector(
                p_props=p_props,
                orientation=orientation,
                snap_time=cur_t,
                max_sliceax_abspos=max_sliceax_abspos,
            )
    else:
        assert preset_name is None

    extent = (
        left_edge[im_x_ind],
        left_edge[im_x_ind] + domain_dims[im_x_ind],
        left_edge[im_y_ind],
        left_edge[im_y_ind] + domain_dims[im_y_ind],
    )

    if fig is None:
        assert ax is None
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    elif ax is None:
        raise RuntimeError("fig and ax must both be None or neither can be None")

    if override_fn is None:
        doFullLogSlicePlot2(
            ax,
            fn(slc_dset, orientation, u_obj, domain_dims=domain_dims),
            slc_particle_selection=slc_particle_selection,
            title=f"{quan_name} at {pretty_t_str}",
            extent=extent,
            orientation=orientation,
            cax=cax,
            **plot_kwargs,
        )
    else:
        override_fn(
            ax=ax,
            slc_dset=slc_dset,
            orientation=orientation,
            u_obj=u_obj,
            domain_dims=domain_dims,
            extent=extent,
        )
    return fig


def _get_hdr_dset_kind(n, dnamein, plot_proj, load_distributed_files=False):
    if plot_proj:
        kind = "proj"
        if load_distributed_files:
            hdr, dset = concat_proj(n, dnamein)
        else:
            path = f"{dnamein}/{n}_proj.h5"
            hdr, dset = load_2D_h5(path)
    else:
        kind = "slice"
        if load_distributed_files:
            hdr, dset = concat_slice(n, dnamein)
        else:
            path = f"{dnamein}/{n}_slice.h5"
            hdr, dset = load_2D_h5(path)
    return hdr, dset, kind


def make_slice_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    dnamein: str,
    n: int,
    preset_name: str,
    orientation: str,
    plot_proj: bool = False,
    load_distributed_files: bool = False,
    override_fn: Any = None,
) -> plt.Figure:
    hdr, dset, kind = _get_hdr_dset_kind(n, dnamein, plot_proj, load_distributed_files)
    return _make_2d_plot(
        hdr,
        dset,
        p_props=None,
        u_obj=ChollaUnits(),
        preset_name=preset_name,
        orientation=orientation,
        particle_selector=None,
        kind=kind,
        fig=fig,
        ax=ax,
        cax=cax,
        override_fn=override_fn,
    )


def itr_orientation_preset(
    orientation: str | Sequence[str], preset_name: str | Sequence[str]
) -> Iterable[tuple[str, str]]:
    """
    Makes an iterable over all specified combinations unique
    `(orientation, preset_name)` pairs
    """
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


def make_slice_plot(
    dnamein: str,
    n: int,
    preset_name: str | Sequence[str],
    orientation: str | Sequence[str],
    callback: Callable[[plt.Figure, int, str, str, str, bool], None],
    plot_proj: bool = False,
    load_distributed_files: bool = False,
    particle_selector: ParticleSelectCallback = None,
):
    p_props = None
    hdr, dset, kind = _get_hdr_dset_kind(n, dnamein, plot_proj, load_distributed_files)
    if particle_selector is not None:
        if load_distributed_files:
            _, p_props = concat_particles(n, dnamein)
        else:
            raise RuntimeError()

    u_obj = ChollaUnits()

    out_l = []
    for orient, pn in itr_orientation_preset(orientation, preset_name):
        fig = _make_2d_plot(
            hdr,
            dset,
            p_props,
            u_obj=ChollaUnits(),
            preset_name=pn,
            orientation=orient,
            particle_selector=particle_selector,
            kind=kind,
        )

        if callback is None:
            out_l.append(fig)
        else:
            out_l.append(callback(orientation=orient, preset_name=pn, fig=fig))

    if isinstance(orientation, str) and isinstance(preset_name, str):
        assert len(out_l) == 1
        return out_l[0]
    else:
        return out_l


def _saver(
    fig: plt.Figure | None,
    n: int | None,
    orientation: str,
    preset_name: str,
    outdir_prefix: str,
    try_makedirs: bool = True,
):
    outdir = f"{outdir_prefix}/{orientation}/{preset_name}/"
    if try_makedirs:
        os.makedirs(outdir, exist_ok=True)
    if (fig is not None) and (n is not None):
        fig.tight_layout()
        plt.savefig(f"{outdir}/{n:04d}.png", dpi=300)
        plt.close(fig)


def make_plot(
    n: int,
    preset_name: str | Sequence[str],
    run_dir: str,
    *,
    outdir_prefix: str,
    orientation: str | Sequence[str] = "xz",
    try_makedirs: bool = False,
    plot_proj: bool = False,
    particle_selector: ParticleSelectCallback | None = None,
    load_distributed_files=False,
) -> int:
    callback = partial(
        _saver, n=n, try_makedirs=try_makedirs, outdir_prefix=outdir_prefix
    )

    make_slice_plot(
        dnamein=run_dir,
        n=n,
        preset_name=preset_name,
        orientation=orientation,
        callback=callback,
        plot_proj=plot_proj,
        particle_selector=particle_selector,
        load_distributed_files=load_distributed_files,
    )
    return n


class _FuncWrapper:
    """
    Wraps a callable function and produces specific exception handling:
    - we pass through a KeyboardInterupt
    - we print other exceptions and return `None`
    """

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
            indent = "   "

            tmp = (
                format_exc().replace("\n", "\n" + indent)
                + "\n"
                + ("\n" + indent).join(format_tb(exc_traceback))
            )
            print(
                f"error encountered during call with args = {args!r}, kwargs = "
                f"{kwargs!r}. The Error is:\n{indent}{tmp}\n",
                flush=True,
            )


def make_plots(
    n_itr: Iterable[int],
    preset_name: str | Sequence[str],
    *,
    run_dir: str,
    plot_proj: bool,
    outdir_prefix: str,
    orientation: str | Sequence[str] = "xz",
    load_distributed_files: bool = False,
    particle_selector: ParticleSelectCallback | None = None,
    pool: PoolType | None = None,
):
    """
    Generates a series of plots

    Parameters
    ----------
    n_itr
        An iterable specifying the snapshot numbers to plot
    preset_name
        A single name or the list of names of all presets that should be plotted
    run_dir
        Path to the directory where the snapshot files are stored
    plot_proj
        Indicates whether we are plotting a projection or a slice
    outdir_prefix
        All resulting snapshots are stored in this directory (several layers are made)
    orientation
        A single name or a list of names of the orientations that should be plotted
    load_distributed_files
        Indicates whether we are loading distributed datasets or a dataset with all
        data in a single file
    particle_selector
        A callback used for selecting/preparing particle data for plotting
    pool
        Optionally specifies a multiprocessing.Pool to parallelize the calculation
    """
    # try to make the output directories ahead of time
    # -> (otherwise, we might waste a lot of time trying to constantly remake them)
    for orient, pn in itr_orientation_preset(orientation, preset_name):
        _saver(
            fig=None,
            n=None,
            orientation=orient,
            preset_name=pn,
            outdir_prefix=outdir_prefix,
            try_makedirs=True,
        )

    func = _FuncWrapper(
        make_plot,
        preset_name=preset_name,
        orientation=orientation,
        try_makedirs=False,
        run_dir=run_dir,
        plot_proj=plot_proj,
        outdir_prefix=outdir_prefix,
        load_distributed_files=load_distributed_files,
        particle_selector=particle_selector,
    )

    my_map = map if pool is None else pool.imap_unordered

    for rslt in my_map(func, n_itr):
        print("done: ", rslt)
