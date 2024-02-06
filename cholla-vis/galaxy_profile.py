import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np
import unyt
import yt

import examination_plot_utils

def _get_override_bins(ds):
    tmp = examination_plot_utils.cell_widths(ds)
    assert tmp[0] == tmp[1]

    rmax = np.sqrt(ds.domain_left_edge[0]**2 + ds.domain_left_edge[1]**2)
    count = np.floor((rmax / tmp[0]).to('dimensionless'))
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

def get_profile(ds,grid, field, weight_field = ('gas','mass')):
    radial_bins, z_bins = _get_override_bins(ds)

    codeL='code_length'
    cylRadius = grid["index", "cylindrical_radius"].to(codeL).v
    cylZ = grid["index", "cylindrical_z"].to(codeL).v

    codeL_uq = ds.units.code_length

    field_arr = grid[field]
    field_uq = field_arr.uq
    field_arr = field_arr.v.flatten()

    weight_arr = grid[weight_field]
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
            print('no positive velocities')
            return
        vmin = H[H>0].min()
        vmax = H.max()
        kwargs['norm'] = LogNorm(vmin,vmax)
    return ax.pcolormesh(X, Y, H.T, **kwargs)

def mydigitize(x, bins):
    x = x.to(bins.units)
    return np.digitize(x.ndview,bins.ndview)

class My2DProfile:
    def __init__(self, x_edges, y_edges, H_dict,
                 x_field = None, y_field = None):
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.H_dict = H_dict
        self.x_field, self.y_field = x_field, y_field

    def show_2D_profile(self, ax, H_name, **kwargs):
        return show_2D_profile(
            ax, self.x_edges,self.y_edges, 
            self.H_dict[H_name], **kwargs
        )

    def __call__(self, H_name, x_vals, y_vals):
        x_ind = mydigitize(x_vals, self.x_edges) - 1
        y_ind = mydigitize(y_vals, self.y_edges) - 1

        # if x_ind is negative, then smaller than the smallest value
        # if y_ind 

        #print(x_ind.min(), x_ind.max(), self.x_edges.shape)
        #print(y_ind.min(), y_ind.max(), self.y_edges.shape)
        #print(self.H_dict[H_name].shape)

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

def add_corrected_vxy_field(profile, ds = None, force_override = True):

    def _corrected_vxy(field, data):
        x_vals = data['index','x']
        y_vals = data['index','y']
        z_vals = data['index','z']

        cyl_r = np.sqrt(x_vals * x_vals + y_vals * y_vals)
        cyl_theta = np.arctan2(y_vals.to(x_vals.units).ndview, 
                               x_vals.ndview)
        cyl_r = data['index','cylindrical_radius']
        avg_vcyltheta = profile(H_name = ('gas', 'velocity_cylindrical_theta'), 
                                x_vals = cyl_r, y_vals = z_vals)
        if field.name[1][-1] == 'x':
            xhat = - np.sin(cyl_theta) # xhat dot thetahat
            return data['gas','velocity_x'] - xhat * avg_vcyltheta
        elif field.name[1][-1] == 'y':
            yhat = np.cos(cyl_theta) # yhat dot thetahat
            return data['gas','velocity_y'] - yhat * avg_vcyltheta

    kwargs = dict(function=_corrected_vxy,
                  sampling_type="local", units="auto",
                  dimensions=unyt.dimensions.velocity,
                  force_override = force_override)
    for field in [('gas', 'corrected_vx'), ('gas', 'corrected_vy')]:
        if ds is not None:
            ds.add_field(field, **kwargs)
        else:
            yt.add_field(field, **kwargs)

def _build_profile(ds,grid, field_l):
    temp = {}
    for field in field_l:
        print(field)
        radial_bins, z_bins, H, total_mass =get_profile(
            ds,grid, field,
            weight_field = ('gas','mass')
        )
        temp[field] = H

    return My2DProfile(
        x_edges = radial_bins, y_edges = z_bins, H_dict = temp,
        x_field = ("index", "cylindrical_radius"),
        y_field = ("index", "cylindrical_z")
    )

def build_profile(path = f'/ix/eschneider/mabruzzo/hydro/cholla/test-prog/norewind2-noAvg-tall/0.h5.0',
                  make_plot = True, field_l = [('gas', 'velocity_cylindrical_theta')]):
    ds = yt.load(path)
    grid = ds.covering_grid(0, left_edge = ds.domain_left_edge, dims = ds.domain_dimensions)

    # build the actual profile
    myprof = _build_profile(ds,grid, field_l)

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