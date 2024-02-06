import numpy as np
import unyt
import yt
from typing import Tuple

def build_covering_grid(ds):
    return ds.covering_grid(0, left_edge = ds.domain_left_edge, dims = ds.domain_dimensions)
    
def _coerce_ds(ds):
    if isinstance(ds, yt.data_objects.construction_data_containers.YTCoveringGrid):
        ds = ds.ds
    assert ds.coordinates.axis_order == ('x','y','z')
    return ds
def cell_widths(ds):
    return (ds.domain_right_edge - ds.domain_left_edge) / ds.domain_dimensions

def get_cell_pos(ds, axis, cell_edges = False):
    ds = _coerce_ds(ds)
    ind = ('x','y','z').index(axis)
    if cell_edges:
        tmp = np.arange(ds.domain_dimensions[ind] + 1)
    else:
        tmp = 0.5 + np.arange(ds.domain_dimensions[ind])
    out = ds.domain_left_edge[ind] + tmp * cell_widths(ds)[ind]
    if cell_edges:
        out[-1] = ds.domain_right_edge[ind]
    return out

def get_containing_cellindex(ds, axis, loc, floored_int = True):
    ds = _coerce_ds(ds)
    ind = ('x','y','z').index(axis)
    loc = loc.to(ds.units.code_length)
    assert (loc >= ds.domain_left_edge[ind]).all() and (loc <= ds.domain_right_edge[ind]).all()
    out = ((loc - ds.domain_left_edge[ind]) / cell_widths(ds)[ind]).to('dimensionless').v
    if floored_int:
        return np.floor(out).astype(np.int64)
    else:
        return out

_axis_names = ('x','y','z')

def index_slcs_from_pairs(arg, x = None, y = None, z =None):
    ds = _coerce_ds(arg)

    def _slc_from_pair(axis_index, pair):
        if pair is None:
            return slice(0,ds.domain_dimensions[axis_index])
        lo,hi = pair
        assert np.size(lo) == 1 and np.ndim(lo) == 0
        assert np.size(hi) == 1 and np.ndim(hi) == 0
        if isinstance(lo, unyt.unyt_array):
            lo = lo.to(ds.units.code_length).v
        if isinstance(hi, unyt.unyt_array):
            hi = hi.to(ds.units.code_length).v
        pair = get_containing_cellindex(ds, _axis_names[axis_index], 
                                        ds.arr([float(lo),float(hi)], 'code_length'), 
                                        floored_int = False)
        return slice(int(pair[0]), int(np.ceil(pair[1])))

    return _slc_from_pair(0, x), _slc_from_pair(1, y), _slc_from_pair(2, z)

def plot_slice_grid_old(ax, grid, coord, coord_val, field, lims = {},
                        set_axis_label = True, func = None, unit = None, **kwargs):
    assert 'extent' not in kwargs

    assert grid.ds.coordinates.axis_order == _axis_names
    ds = grid.ds

    if coord in lims or not all(k in _axis_names for k in lims):
        raise RuntimeError("lims can only contain limits for x,y,z")

    if isinstance(coord_val, int):
        ind = coord_val
    else:
        ind = get_containing_cellindex(ds, axis = coord, loc = coord_val)

    if coord == 'x':
        im_x_label,im_y_label = ('$y$ [code length]', '$z$ [code length]')
        _, y_slc, z_slc = index_slcs_from_pairs(ds, **lims)
        idx = (ind, y_slc, z_slc)
        y_edges = get_cell_pos(ds, 'y', cell_edges = True)
        z_edges = get_cell_pos(ds, 'z', cell_edges = True)

        # I don't think we need to add 1 to slc.stop
        imshow_extent = (y_edges[y_slc.start].v, y_edges[y_slc.stop].v,
                         z_edges[z_slc.start].v, z_edges[z_slc.stop].v)
    elif coord == 'y':
        raise RuntimeError()
    elif coord == 'z':
        im_x_label,im_y_label = ('$x$ [code length]', '$y$ [code length]')
        x_slc,y_slc,_ = index_slcs_from_pairs(ds, **lims)
        idx = (x_slc, y_slc, ind)

        x_edges = get_cell_pos(ds, 'x', cell_edges = True)
        y_edges = get_cell_pos(ds, 'y', cell_edges = True)

        # I don't think we need to add 1 to slc.stop
        imshow_extent = (x_edges[x_slc.start].v, x_edges[x_slc.stop].v,
                         y_edges[y_slc.start].v, y_edges[y_slc.stop].v)

    if func is None:
        vals = grid[field][idx]
    else:
        vals = func(grid,idx)
    if unit is not None:
        vals = vals.to(unit)
    img = ax.imshow(vals.T, extent = imshow_extent, interpolation = 'none', 
                     origin = 'lower', **kwargs)

    extrema = (vals.min(), vals.max())
    if set_axis_label:
        ax.set_xlabel(im_x_label)
        ax.set_ylabel(im_y_label)
    return img, extrema

from typing import Tuple

def _come_up_with_grid_edges(ds, coord, coord_val, lims = {}):
    assert ds.coordinates.axis_order == _axis_names

    if isinstance(coord_val, int):
        ind = np.array([coord_val])
    elif np.ndim(coord_val) == 0:
        ind = np.array([get_containing_cellindex(ds, axis = coord, loc = coord_val)])
    else:
        assert np.ndim(coord_val) == 1
        if isinstance(coord_val[0], int):
            ind = np.array(coord_val)
        else:
            ind = np.array([get_containing_cellindex(ds, axis = coord, loc = e) for e in coord_val])

    x_edges = get_cell_pos(ds, 'x', cell_edges = True)
    y_edges = get_cell_pos(ds, 'y', cell_edges = True)
    z_edges = get_cell_pos(ds, 'z', cell_edges = True)

    u = 'code_length'
    if coord == 'x':
        im_x_label,im_y_label = ('$y$ [code length]', '$z$ [code length]')
        _, y_slc, z_slc = index_slcs_from_pairs(ds, **lims)
        idx = (ind, y_slc, z_slc)

        left_edge = ds.arr(
            [x_edges[ind.min()].to(u).v, y_edges[y_slc.start].to(u).v, z_edges[z_slc.start].to(u).v],
            u)
        # I don't think we need to add 1 to slc.stop
        right_edge = ds.arr(
            [x_edges[ind.max()].to(u).v, y_edges[y_slc.stop].to(u).v, z_edges[z_slc.stop].to(u).v],
            u)
        shape = (ind.max() + 1 - ind.min(), y_slc.stop - y_slc.start, z_slc.stop - z_slc.start)

        leftmost_index = (ind.max(), y_slc.start, z_slc.start)
    elif coord == 'y':
        raise RuntimeError()
    elif coord == 'z':
        im_x_label,im_y_label = ('$x$ [code length]', '$y$ [code length]')
        x_slc,y_slc,_ = index_slcs_from_pairs(ds, **lims)
        idx = (x_slc, y_slc, ind)

        left_edge = ds.arr(
            [x_edges[x_slc.start].to(u).v, y_edges[y_slc.start].to(u).v, z_edges[ind.min()].to(u).v],
            u)
        # I don't think we need to add 1 to slc.stop
        right_edge = ds.arr(
            [x_edges[x_slc.stop].to(u).v, y_edges[y_slc.stop].to(u).v, z_edges[ind.max()].to(u).v],
            u)
        shape = (x_slc.stop - x_slc.start, y_slc.stop - y_slc.start, ind.max() + 1 - ind.min())

        leftmost_index = (x_slc.start, y_slc.start, ind.min())

    return left_edge, right_edge, shape, leftmost_index


class GridPlotDetails:
    left_edge: unyt.unyt_array
    right_edge: unyt.unyt_array
    shape: Tuple[int,int,int]
    leftmost_index: Tuple[int,int,int]

    def __init__(self, left_edge, right_edge, shape, leftmost_index):
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.shape = shape
        self.leftmost_index = leftmost_index

    @classmethod
    def build_for_slice_grids(cls, ds, coord, coord_val, lims = {}):
        left_edge, right_edge, shape, leftmost_index = _come_up_with_grid_edges(
            ds, coord, coord_val, lims = lims)

        if not (ds.domain_left_edge <= left_edge).all():
            raise RuntimeError(
                f'the calculated grid left-edge is {left_edge!r} while the '
                f'domain_left_edge is {ds.domain_left_edge!r}')
        elif not (ds.domain_right_edge >= right_edge).all():
            raise RuntimeError(
                f'the calculated grid right-edge is {right_edge!r} while the '
                f'domain_right_edge is {ds.domain_right_edge!r}')
        return cls(left_edge, right_edge, shape, leftmost_index)

    def construct_grid(self, ds):
        # there seems to be an annoying bug when an entry of shape is 1. In this case, we make it 2
        # but lets do some error checking to explain what happens
        is_right_aligned = np.abs((ds.domain_right_edge - self.right_edge)) < 0.5 * cell_widths(ds)
        for i in range(3):
            if (self.shape[i] == 1) and is_right_aligned[i]:
                raise ValueError("There is a problem constucting a grid when the size along an axis is 1 "
                                 "and that axis is flush against the right edge of the domain. This "
                                 f"problem occurs for axis: {i}") 
        shape = tuple([max(e,2) for e in self.shape])

        out = ds.covering_grid(0, left_edge = self.left_edge, dims = shape)
        return out

    def check_grid_consistency(self, grid):
        # in principle, we may want to return the slice indices that would give consistent results
        # in the scenario where we read the full grid!
        widths = cell_widths(grid.ds)

        if self.shape != grid.shape:
            return False
        elif np.any(np.abs(self.left_edge - grid.LeftEdge) > widths):
            return False
        elif np.any(np.abs(self.right_edge - grid.RightEdge) > widths):
            return False
        return True

    def idx_to_specified_grid_idx(self, idx, grid):
        widths = cell_widths(grid.ds)
        left_edge_cell_offsets = (
            (np.abs(self.left_edge - grid.LeftEdge) / widths).to('dimensionless').v
        )
        num_shift = (left_edge_cell_offsets + 0.5).astype(int)
        if np.any(left_edge_cell_offsets <= -0.5):
            raise ValueError(f"The left edge of grid, {grid.LeftEdge}, lies to the right "
                             f"of self.left_edge, {self.left_edge}")
        elif np.all((grid.RightEdge - self.right_edge) < -0.5 * widths):
            raise ValueError(f"The right edge of grid, {grid.RightEdge}, lies to the left "
                             f"of self.right_edge, {self.right_edge}")

        nominal_slc_len = self.shape
        nominal_slc_start = self.leftmost_index

        def update_slice(slc, i):
            if slc.start is None:
                start = nominal_slc_start[i] + num_shift[i]
            else:
                start = nominal_slc_stop
            if slc.stop is None:
                stop = start + nominal_slc_len[i]
            else:
                stop = slc.stop + num_shift[i]
            return slice(start, stop, slc.step)
        out = [None, None, None]
        for i in range(3):
            is_slice = isinstance(idx[i], slice)
            if isinstance(idx[i], slice):
                out[i] = update_slice(idx[i], i)
            else:
                out[i] = idx[i] + num_shift[i]
        return tuple(out)



def plot_slice_grid2(ax, ds, coord, coord_val, field, lims = {},
                     set_axis_label = True, func = None, unit = None,
                     reuse_grid = None, **kwargs):
    assert 'extent' not in kwargs

    grid_details = GridPlotDetails.build_for_slice_grids(ds, coord, coord_val, lims)

    if coord in lims or not all(k in _axis_names for k in lims):
        raise RuntimeError("lims can only contain limits for x,y,z")

    if isinstance(coord_val, int):
        ind = coord_val
    else:
        ind = get_containing_cellindex(ds, axis = coord, loc = coord_val)

    if coord == 'x':
        im_x_label,im_y_label = ('$y$ [code length]', '$z$ [code length]')
        y_slc, z_slc = slice(None), slice(None)
        idx = (ind - grid_details.leftmost_index[0], y_slc, z_slc)

        imshow_extent = (grid_details.left_edge[1].v, grid_details.right_edge[1].v,
                         grid_details.left_edge[2].v, grid_details.right_edge[2].v)
    elif coord == 'y':
        raise RuntimeError()
    elif coord == 'z':
        im_x_label,im_y_label = ('$x$ [code length]', '$y$ [code length]')
        x_slc,y_slc = slice(None), slice(None)
        idx = (x_slc, y_slc, ind - grid_details.leftmost_index[0])

        # I don't think we need to add 1 to slc.stop
        imshow_extent = (grid_details.left_edge[0].v, grid_details.right_edge[0].v,
                         grid_details.left_edge[1].v, grid_details.right_edge[1].v)

    if reuse_grid is None:
        grid = grid_details.construct_grid(ds)
    else:
        grid = reuse_grid
        idx = grid_details.idx_to_specified_grid_idx(idx, grid)

    if func is None:
        vals = grid[field][idx]
    else:
        vals = func(grid,idx)
    if unit is not None:
        vals = vals.to(unit)
    img = ax.imshow(vals.T, extent = imshow_extent, interpolation = 'none', 
                     origin = 'lower', **kwargs)

    extrema = (vals.min(), vals.max())
    if set_axis_label:
        ax.set_xlabel(im_x_label)
        ax.set_ylabel(im_y_label)
    return img, extrema, grid