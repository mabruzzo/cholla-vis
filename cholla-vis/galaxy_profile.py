import functools
from typing import Any, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np
import unyt
import yt

import examination_plot_utils
from utils import PseudoDS


_ABSZ_FIELD = ('index', 'absz')

def try_add_absz(ds):
    def _abs_z_field(field, data):
        return np.abs(data[('index', 'z')])
    field_name = _ABSZ_FIELD
    if field_name not in ds.derived_field_list:
        ds.add_field(
            field_name,
            function=_abs_z_field,
            sampling_type="local",
            units="auto",
            dimensions="length"
        )

def try_add_extra_fields(ds):

    _outwardz_velocity_field = ("gas", "outwardz_velocity")
    def _outwardz_velocity(field, data):
        out = np.sign(data["index", "z"]) * data["gas", "velocity_z"]
        out.convert_to_units('code_velocity') # convert in place
        return out

    def _has_positive_outwardz_velocity(field, data):
        out = (data[_outwardz_velocity_field].ndview > 0)
        return unyt.unyt_array(out.astype(np.float64), 'dimensionless')

    def _has_positive_velocity_spherical_radius(field, data):
        out = (data['gas', 'velocity_spherical_radius'].ndview > 0)
        return unyt.unyt_array(out.astype(np.float64), 'dimensionless')

    adiabat_K_units = 'cm**4/(g**(2/3)*s**2)'
    def _adiabat_K(field, data):
        if np.shape(ds.gamma) == (1,):
            gamma = ds.gamma[0]
        else:
            gamma = ds.gamma
        assert np.abs((ds.gamma - (5/3)) / (5/3)) < 1e-8
        tmp = (
            data['gas', 'pressure'].in_cgs().ndview /
            (data['gas', 'density'].in_cgs().ndview)**(5/3)
        )
        return unyt.unyt_array(tmp, adiabat_K_units)

    def _mass_zsq(field, data):
        return (
            data["gas", "cell_mass"] * np.square(data["index", "z"])
        ).to("code_mass*code_length**2")

    def _massT(field, data):
        return (
            data["gas", "cell_mass"] * data["gas", "temperature"]
        ).to("code_mass*K")

    def _mass_vcirc(field, data):
        return (
            data["gas", "cell_mass"] * data["gas", "velocity_cylindrical_theta"]
        ).to("code_mass*code_length/code_time")

    def _symmetric_theta(field, data):
        # basically we are taking the absolute value
        ratio = (data[_ABSZ_FIELD]/data['index', 'spherical_radius'])
        ratio.convert_to_units('dimensionless') # convert in place
        return unyt.unyt_array(np.arccos(ratio.ndview),'radian')

    def _pturb_z(field, data):
        out = np.square(data["gas", "velocity_z"])
        out *= data["gas", "density"]
        out.convert_to_units("dyne/cm**2") # convert in place
        return out

    def _ptot_slab(field, data):
        # the total pressure in a slab-geometry
        # -> inspired by Ostriker & Kim 2022
        #    (while their definition of thermal pressure looks weird, I'm pretty sure
        #    that's because they have dropped gamma from their sound speed definition
        #    as in the original TIGRESS paper)
        out = _pturb_z(field, data)
        out += data["gas", "pressure"]
        out.convert_to_units("dyne/cm**2") # convert in place
        return out

    triples = [
        (_outwardz_velocity_field, _outwardz_velocity, "code_velocity"),
        (("index", "has_positive_outwardz_velocity"),
         _has_positive_outwardz_velocity, "dimensionless"),
        (("index", "has_positive_velocity_spherical_radius"),
         _has_positive_velocity_spherical_radius, "dimensionless"),
        (("gas", "adiabat"), _adiabat_K, adiabat_K_units),
        (("gas", "mass_zsq"), _mass_zsq, "code_mass*code_length**2"),
        (("gas", "massT"), _massT, "code_mass*K"),
        (("gas", "mass_vcirc"), _mass_vcirc, "code_mass*code_length/code_time"),
        (("index", "symmetric_theta"), _symmetric_theta, "radian"),
        (("gas", "pressure_tot_slab"), _ptot_slab, "dyne/cm**2")
    ]
    for field_name, fn, units in triples:
        if field_name not in ds.derived_field_list:
            ds.add_field(
                field_name,
                function=fn,
                force_override=True,
                sampling_type="local",
                units=units,
            )

def add_flux_fields(ds, radial):
    if radial:
        vfield = ('gas', 'velocity_spherical_radius')
        _rho_flux_field = ("gas", "flux_rho_spherical_radius")
        _momentum_flux_field = ("gas", "flux_momentum_spherical_radius")
        _e_flux_field = ("gas", "flux_e_spherical_radius")
    else:
        vfield = ("gas", "outwardz_velocity")
        _rho_flux_field = ("gas", "flux_rho_outwardz")
        _momentum_flux_field = ("gas", "flux_momentum_outwardz")
        _e_flux_field = ("gas", "flux_e_outwardz")

    def _rho_flux(field, data):
        rho = data["gas", "density"]
        vel = data[vfield]
        return rho * vel

    def _momentum_flux(field, data):
        rho = data["gas", "density"]
        vel = data[vfield]
        p = data["gas", "pressure"]
        return rho * vel*vel + p

    gamma = ds.gamma
    gamma_div_gm1 = gamma/(gamma-1)
    def _e_flux(field, data):
        rho = data["gas", "density"]
        vel = data[vfield]
        p = data["gas", "pressure"]

        v_sq = (
            np.square(data['gas', 'velocity_x']) +
            np.square(data['gas', 'velocity_y']) +
            np.square(data['gas', 'velocity_z'])
        )

        bernoulli = (
            (0.5 * v_sq) +
            gamma_div_gm1 * (p / rho)
        )
        return rho * vel * bernoulli

    field_sets = [
        (_rho_flux_field, _rho_flux,
         unyt.dimensions.velocity * unyt.dimensions.density),
        (_momentum_flux_field, _momentum_flux,
         unyt.dimensions.mass / (unyt.dimensions.length * unyt.dimensions.time**2)),
        (_e_flux_field, _e_flux,
         unyt.dimensions.velocity**3 * unyt.dimensions.density)
    ]
    for field_name, fn, dimensions in field_sets:
        if field_name not in ds.derived_field_list:
            ds.add_field(
                field_name,
                function=fn,
                force_override=True,
                sampling_type="local",
                units="auto",
                dimensions=dimensions
            )

def _get_override_bins(ds):
    tmp = examination_plot_utils.cell_widths(ds)
    assert tmp[0] == tmp[1]

    rmax = np.sqrt(ds.domain_left_edge[0]**2 + ds.domain_left_edge[1]**2)
    count = np.floor((rmax / tmp[0])).to('dimensionless').v
    if (count*tmp[0]) > rmax:
        radial_bins = np.arange(count+1)*tmp[0]
    elif ((count+1)*tmp[0]) > rmax:
        radial_bins = np.arange(count+2)*tmp[0]
    else:
        radial_bins = np.arange(count+3)*tmp[0]
    assert radial_bins[-1] > rmax
    assert radial_bins[-2] <= rmax

    z_edges = examination_plot_utils.get_cell_pos(
        ds, axis = 'z', cell_edges = True
    )
    #z_bins = z_edges[z_edges >= ds.quan(0,'code_length')]
    z_bins = z_edges
    return radial_bins, z_bins

def get_profile(profile_arg, field, weight_field = ('gas','mass'), bin_dicts = None):
    if isinstance(profile_arg, PseudoDS):
        data_src = profile_arg
        dflt_radial_bins, dflt_z_bins = _get_override_bins(data_src.domain_props)
        codeL='kpc'
        codeL_uq = unyt.kpc
    else:
        ds, grid = profile_arg
        data_src = grid
        dflt_radial_bins, dflt_z_bins = _get_override_bins(ds)
        codeL='code_length'
        codeL_uq = ds.units.code_length
    if bin_dicts is None:
        bin_dicts = {}

    radial_bins = bin_dicts.get(("index", "cylindrical_radius"), dflt_radial_bins).copy()
    z_bins = bin_dicts.get(("index", "cylindrical_z"), dflt_z_bins).copy()
    
    cylRadius = data_src["index", "cylindrical_radius"].to(codeL).v
    cylZ = data_src["index", "cylindrical_z"].to(codeL).v

    field_arr = data_src[field]
    field_uq = field_arr.uq
    field_arr = field_arr.v.flatten()

    if weight_field is None:
        weight_uq = 1.0
        weight_arr = np.broadcast_to(1.0, field_arr.shape)
    else:
        weight_arr = data_src[weight_field]
        weight_uq = weight_arr.uq
        weight_arr = weight_arr.v.flatten()

    weighted_sum, _, _ = np.histogram2d(
        x = cylRadius.flatten(),
        y = cylZ.flatten(),
        bins = (radial_bins.to(codeL).v,
                z_bins.to(codeL).v),
        density = False,
        weights = weight_arr*field_arr
    )

    weight_sum, _, _ = np.histogram2d(
        x = cylRadius.flatten(),
        y = cylZ.flatten(),
        bins = (radial_bins.to(codeL).v,
                z_bins.to(codeL).v),
        density = False,
        weights = weight_arr
    )

    H_field = np.empty_like(weighted_sum)
    w = (weight_sum > 0)
    H_field[w] = (weighted_sum[w] / weight_sum[w])
    H_field[~w] = np.nan
    H_field *= field_uq

    H_weight = weight_sum * weight_uq
    # x,y, h_field, H_weight
    return radial_bins, z_bins, H_field, H_weight



def mydigitize(x, bins):
    x = x.to(bins.units)
    return np.digitize(x.ndview,bins.ndview)

class My1DProfile:
    def __init__(self, edges, data):
        self.edges, self.data = edges,data

    def __call__(self, x):
        ind = mydigitize(x, self.edges) - 1
        w = np.logical_and(x >= self.edges.min(),
                           x < self.edges.max())
        if w.all():
            return self.data[ind]
        else:
            out = unyt.unyt_array(
                np.empty(shape = x.shape, dtype = 'f8'),
                self.data.units
            )
            out.ndview[w] = self.data.ndview[ind[w]]
            out.ndview[~w] = np.nan
            return out

class My2DProfile:
    def __init__(self, x_edges, y_edges, H_dict,
                 x_field = None, y_field = None):
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.H_dict = H_dict
        self.x_field, self.y_field = x_field, y_field

    def show_contour(self, ax, H_name, levels = None, H_func = None,
                     **kwargs):
        x = 0.5*(self.x_edges[1:] + self.x_edges[:-1])
        y = 0.5*(self.y_edges[1:] + self.y_edges[:-1])

        args = ()
        if levels is not None:
            args = (levels,)

        H = self.H_dict[H_name]
        if H_func is not None:
            H = H_func(H)
        if isinstance(H, unyt.unyt_array):
            H = H.ndview
        return ax.contour(x, y, H.T, *args, **kwargs)

    def show_2D_profile(self, ax, H_name, H_func = None, **kwargs):
        if H_func is None:
            H_func = lambda H: H
        return show_2D_profile(
            ax, self.x_edges,self.y_edges, 
            H_func(self.H_dict[H_name]), **kwargs
        )

    def bin_grid(self, *, axis, edges = True): # could be faster
        try:
            index = {'x' : 0, 'y' : 1, 'both' : slice(0,2,1)}[axis]
        except KeyError:
            raise ValueError('axis must be "x", "y", or "both"') from None 
        if edges:
            return np.meshgrid(self.x_edges, self.y_edges)[index]
        else:
            x = 0.5*(self.x_edges[:-1] + self.x_edges[1:])
            y = 0.5*(self.y_edges[:-1] + self.y_edges[1:])
            return np.meshgrid(x, y)[index]

    def __call__(self, H_name, x_vals, y_vals):
        x_ind = mydigitize(x_vals, self.x_edges) - 1
        y_ind = mydigitize(y_vals, self.y_edges) - 1

        # if x_ind is negative, then smaller than the smallest value
        # if y_ind

        w_x = np.logical_and(x_vals >= self.x_edges.min(),
                             x_vals < self.x_edges.max())
        w_y = np.logical_and(y_vals >= self.y_edges.min(),
                             y_vals < self.y_edges.max())
        w = np.logical_and(w_x,w_y)

        if w.all():
            return self.H_dict[H_name][x_ind, y_ind]
        else:
            out = unyt.unyt_array(
                np.empty(shape = x_vals.shape, dtype = 'f8'),
                self.H_dict[H_name].units
            )
            out.ndview[w] = self.H_dict[H_name].ndview[x_ind[w], y_ind[w]]
            out.ndview[~w] = np.nan
            return out

    def H_names(self):
        return list(self.H_dict.keys())

    def new_profile_from_ratio(self, out_field_name, num_field, denom_field):
        return My2DProfile(
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            H_dict = {out_field_name : self.H_dict[num_field]/self.H_dict[denom_field]},
            x_field = self.x_field,
            y_field = self.y_field
        )

    def averaged_1D_profile(self, y_index, num_field, denom_field):
        numerator = self.H_dict[num_field]
        denominator = self.H_dict[denom_field]
        if isinstance(y_index, int):
            numerator = numerator[:, y_index]
            denominator = denominator[:, y_index]
        else:
            assert y_index is not None
            numerator = numerator[:,y_index].sum(axis=1)
            denominator = denominator[:,y_index].sum(axis=1)
        return My1DProfile(self.x_edges, numerator/denominator)

def add_corrected_vxy_field(profile, ds = None,
                            field_prefix = "corrected",
                            vrot_comp_prefix = 'avgvrot',
                            force_override = True,
                            use_abs_z = False):

    def _component_from_vrot(field, data):
        x_vals = data['index','x']
        y_vals = data['index','y']
        if use_abs_z:
            z_vals = data[_ABSZ_FIELD]
        else:
            z_vals = data['index','z']

        cyl_r = np.sqrt(x_vals * x_vals + y_vals * y_vals)
        cyl_theta = np.arctan2(y_vals.to(x_vals.units).ndview, 
                               x_vals.ndview)
        cyl_r = data['index','cylindrical_radius']
        if isinstance(profile, My2DProfile):
            avg_vcyltheta = profile(H_name = ('gas', 'velocity_cylindrical_theta'), 
                                    x_vals = cyl_r, y_vals = z_vals)
        elif isinstance(profile, My1DProfile):
            avg_vcyltheta = profile(cyl_r)
        else:
            raise RuntimeError()

        if field.name[1][-1] == 'x':
            xhat = - np.sin(cyl_theta) # xhat dot thetahat
            return xhat * avg_vcyltheta
        elif field.name[1][-1] == 'y':
            yhat = np.cos(cyl_theta) # yhat dot thetahat
            return yhat * avg_vcyltheta
        else:
            raise RuntimeError()

    def _corrected_vxy(field, data):
        if field.name[1][-1] == 'x':
            vx_rot = _component_from_vrot(field, data)
            return data['gas','velocity_x'] - vx_rot
        elif field.name[1][-1] == 'y':
            vy_rot = _component_from_vrot(field, data)
            return data['gas','velocity_y'] - vy_rot
        else:
            raise RuntimeError()
    if use_abs_z:
        try_add_absz(ds)

    kwargs = dict(sampling_type="local", units="auto",
                  dimensions=unyt.dimensions.velocity,
                  force_override = force_override)
    for field in [('gas', f'{field_prefix}_vx'), ('gas', f'{field_prefix}_vy')]:
        if ds is not None:
            ds.add_field(field, function=_corrected_vxy, **kwargs)
        else:
            yt.add_field(field, function=_corrected_vxy, **kwargs)

    for field in [('gas', f'{vrot_comp_prefix}_vx'), ('gas', f'{vrot_comp_prefix}_vy')]:
        if ds is not None:
            ds.add_field(field, function=_component_from_vrot, **kwargs)
        else:
            yt.add_field(field, function=_component_from_vrot, **kwargs)

def _build_profile(profile_arg, field_l, weight_field, bin_dicts):
    temp = {}
    for field in field_l:
        print(field)
        radial_bins, z_bins, H, total_mass =get_profile(
            profile_arg, field,
            weight_field = weight_field,
            bin_dicts = bin_dicts
        )
        temp[field] = H

    return My2DProfile(
        x_edges = radial_bins, y_edges = z_bins, H_dict = temp,
        x_field = ("index", "cylindrical_radius"),
        y_field = ("index", "cylindrical_z")
    )

def build_profile(data, make_plot = True, field_l = [('gas', 'velocity_cylindrical_theta')],
                  weight_field = ('gas','mass'), bin_dicts = None):
    if isinstance(data,str):
        ds = yt.load(data)
        grid = ds.covering_grid(0, left_edge = ds.domain_left_edge, dims = ds.domain_dimensions)
        profile_arg = (ds,grid)
    else:
        assert isinstance(data, PseudoDS)
        profile_arg = data

    # build the actual profile
    myprof = _build_profile(profile_arg, field_l, weight_field = weight_field,
                            bin_dicts = bin_dicts)

    # now, make the plot!
    plot_out = (None, None)

    if make_plot:
        from mpl_toolkits.axes_grid1 import AxesGrid

        fig = plt.figure(figsize = (5,6))
        axgrid = AxesGrid(
            fig, (0.085, 0.085, 0.83, 0.83), nrows_ncols=(3, 1),
            axes_pad=0.1, label_mode="L", share_all=True, cbar_location="right",
            cbar_mode="each", cbar_size="5%", cbar_pad="2%", aspect=False,)

        use_log_norm_map = {('gas', 'velocity_cylindrical_radius') : False,
                            ('gas', 'velocity_cylindrical_theta') : True,
                            ('gas', 'velocity_cylindrical_z') : False}

        for i, field in enumerate(field_l):
            lognorm = use_log_norm_map.get(field,False)
            im = myprof.show_2D_profile(axgrid[i].axes, H_name = field, lognorm = lognorm)
            plt.colorbar(im, cax = axgrid.cbar_axes[i], label = field[1])
        plot_out = (fig,axgrid)
    return myprof, plot_out



class _Prop(NamedTuple):
    name: str
    field: Tuple[str, str]
    bins: Any
    range_arg: Any

    def get_range(self, ds):
        if self.range_arg is not None:
            return self.range_arg
        assert ds.coordinates.axis_order == ('x', 'y', 'z')
        if self.name == 'rcyl':
            tmp = np.maximum(np.abs(ds.domain_left_edge[:2]),
                             np.abs(ds.domain_right_edge[:2]))
            return [0.0, np.linalg.norm(tmp.ndview)]
        elif self.name == 'z':
            pair = [ds.domain_left_edge[2].ndview,
                    ds.domain_right_edge[2].ndview]
            if self.field == _ABSZ_FIELD:
                return [0.0, np.amax(np.abs(pair))]
            return pair
        raise RuntimeError("should be unreachable")

def _create_galaxy_profile_binkwargs(
    ds, bins_rcyl = 64, bins_z = 64,
    range_rcyl = None, range_z = None,
    use_abs_z = False,
    extra_field_bins_pair = None
):

    import operator
    override_bins = {}

    if use_abs_z:
        try_add_absz(ds)
        z_field = _ABSZ_FIELD
    else:
        z_field = ('index', 'z')

    props = [
        _Prop('rcyl', ('index', 'cylindrical_radius'), bins_rcyl, range_rcyl),
        _Prop('z', z_field, bins_z, range_z),
    ]

    for prop in props:
        bins_ndim = np.ndim(prop.bins)
        if bins_ndim > 1:
            raise ValueError(
                f"bins_{prop.name} must be a scalar or a 1D array"
            )
        elif (bins_ndim == 1) and np.size(bins) < 2:
            raise ValueError(
                f"when bins_{prop.name} is a 1D array, it must contain at "
                "least 2 entries"
            )
        elif (bins_ndim == 1) and (prop.range_arg is not None):
            raise ValueError(
                f"when bins_{prop.name} is a 1D array, range_{prop.name} "
                "must not be specified")
        elif (bins_ndim == 1):
            override_bins[bin_field] = prop.bins
        else:
            try: # logic borrowed from numpy
                bins_count = operator.index(prop.bins)
            except:
                raise TypeError(
                    f"bins_{prop.name} must be an integer or 1D array"
                )
            if bins_count < 1:
                raise ValueError(
                    f"when bins_{prop.name} is an integer, it must be "
                    "positive"
                )
            range_arg = prop.get_range(ds)
            assert range_arg[0] < range_arg[1]
            override_bins[prop.field] = np.linspace(
                range_arg[0],  range_arg[1], num = bins_count + 1
            )

    out = dict(
        bin_fields = [p.field for p in props],
        units = {p.field : 'code_length' for p in props},
        override_bins = override_bins
    )
    if extra_field_bins_pair is not None:
        assert len(extra_field_bins_pair) == 2
        name, override_bins = extra_field_bins_pair
        assert name in ds.derived_field_list
        assert isinstance(override_bins, unyt.unyt_array)
        out['bin_fields'].append(name)
        out['units'][name] = str(override_bins.units)
        out['override_bins'][name] = override_bins.ndview
    return out

def build_galaxy_profile(ds, fields, data_source = None, weight_field = None, bins_rcyl = 64, bins_z = 64,
                         range_rcyl = None, range_z = None, use_abs_z = False, return_ytprof = False):
    
    kwargs = _create_galaxy_profile_binkwargs(
        ds,
        bins_rcyl=bins_rcyl,
        bins_z=bins_z,
        range_rcyl=range_rcyl,
        range_z=range_z,
        use_abs_z=use_abs_z
    )

    if data_source is None:
        data_source = ds.all_data()

    prof = yt.create_profile(
        data_source,
        fields = fields,
        weight_field = weight_field,
        **kwargs
    )

    out = My2DProfile(x_edges = prof.x_bins, y_edges = prof.y_bins,
                      x_field = prof.x_field, y_field = prof.y_field,
                      H_dict = dict((field,prof[field]) for field in fields))
    if return_ytprof:
        return out, prof
    else:
        return out

_UNSPECIFIED = object()

def show_yt_prof(ax, prof, data_field, imxfield=None, imyfield=None,
                 imzfield_idx = _UNSPECIFIED, lognorm = False,
                 **kwargs):
    available = {
        prof.x_field : (prof.x_bins, 0),
        prof.y_field : (prof.y_bins, 1),
    }
    if hasattr(prof, 'z_field'):
        available[prof.z_field] = (prof.z_bins, 2)

    xedges, imx_curaxis = available[imxfield]
    yedges, imy_curaxis = available[imyfield]

    if len(available) == 2 and imzfield_idx is not _UNSPECIFIED:
        raise ValueError()
    elif len(available) == 3 and imzfield_idx is _UNSPECIFIED:
        raise ValueError()
    elif imxfield == imyfield:
        raise ValueError()

    if len(available) == 3:
        idx = []
        for i in range(3):
            if (i == imx_curaxis) or (i == imy_curaxis):
                idx.append(slice(None))
            else:
                idx.append(imzfield_idx)
                imz_curaxis = i
        C = prof[data_field][tuple(idx)]
        assert C.ndim == 2
        if imx_curaxis > imz_curaxis: imx_curaxis-=1
        if imy_curaxis > imz_curaxis: imy_curaxis-=1
    else:
        C = prof[data_field]

    if imx_curaxis != 0:
        C = C.T


    return show_2D_profile(ax,xedges,yedges, H=C, lognorm=lognorm,
                           **kwargs)

def show_2D_profile(ax,xedges,yedges, H, lognorm = False,
                    **kwargs):
    X, Y = np.meshgrid(xedges.v, yedges.v)
    
    H = H.v

    if lognorm:
        if np.isnan(H).any():
            H = np.copy(H)
            H[np.isnan(H)] = 0
        w = H > 0
        if not w.any():
            print('no positive values')
            return
        vmin = H[H>0].min()
        vmax = H.max()
        kwargs['norm'] = LogNorm(vmin,vmax)
    return ax.pcolormesh(X, Y, H.T, **kwargs)
    
