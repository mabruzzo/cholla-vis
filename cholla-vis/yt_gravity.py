# defining functionality associated with reading the gravitational potential in as a field
# and computing fields that store useful derived quantities

import yt

from functools import partial
import os.path

import h5py
import deriv

def _momentum_cylindrical_phi(field, data):
    mom_x = data['cholla', 'momentum_x']
    mom_y = data['cholla', 'momentum_y']
    theta = data['index', 'cylindrical_theta']
    return -np.sin(theta) * mom_x + np.cos(theta) * mom_y

def _find_potential_gdepth_activeidx(active_zone_shape, size):
    for possible_depth in range(0,4):
        size_guess = np.prod(np.array(active_zone_shape) + 2 * possible_depth)
        if (size_guess == size) and possible_depth == 0:
            return 0, tuple(active_zone_shape), tuple(slice(None) for i in (0,1,2))
        elif (size_guess == size):
            gdepth = possible_depth
            shape_with_ghosts = tuple(active_zone_shape[i] + 2*gdepth
                                      for i in (0,1,2))
            active_idx = tuple(slice(gdepth, -1*gdepth) for i in (0,1,2))
            return gdepth, shape_with_ghosts, active_idx
    raise RuntimeError(
        "There was an issue determining the ghost-depth of the potential")

class PotentialFieldLoader:
    """
    This class acts as a configurable function for loading in the 
    le functionl associated with a simulation.

    By default, this will load in the exact gravity data written out
    with the rest of the data of the snapshot held by the yt-dataset 
    for which this is constructing field data.

    Parameters
    ----------
    fixed_dirname : str, optional
        Specifies the directory holding the gravitational potential data
        (only overwrite this if you want to load data from a different
        simulation.
    fixed_snapshot_prefix : str, optional
        This can be used to force the class from a particular snapshot
    """

    def __init__(self, *, fixed_dirname = None,
                 fixed_snapshot_prefix = None):
        self._fixed_dirname = fixed_dirname
        self._fixed_snapshot_prefix = None

    def construct_path(self, data):
        # data is nominally the "data-argument" of the standard
        # field, data arguments that yt expects to pass to the function
        # specified when deriving fields

        dirname, basename = os.path.split(data.filename)
        prefix, suffix = basename.split('.h5')

        if self._fixed_dirname is not None:
            dirname = self._fixed_dirname

        if self._fixed_snapshot_prefix is not None:
            prefix = self._fixed_snapshot_prefix

        return f'{dirname}/{prefix}_gravity.h5{suffix}'

    def __call__(self, field, data):
        if isinstance(data, yt.fields.api.FieldDetector):
            block_index = 0
            return ds.arr(np.ones(data.shape, dtype = 'f8'),
                                  'code_specific_energy')    

        active_zone_shape = tuple(data.shape)
        with h5py.File(path, 'r') as f:
            potential_dset = f['potential']
            gdepth, shape_with_ghosts, active_idx = \
                _find_potential_gdepth_activeidx(active_zone_shape,
                                                 potential_dset.size)

            assert potential_dset.ndim == 1

            vals = potential_dset[:] # load the full 1D array
            # adjust the shape of vals
            vals.shape = shape_with_ghosts[::-1]
            vals = np.swapaxes(vals, 0,2)
            # now clip the ghost-zones
            out = ds.arr(vals[active_idx], 'code_specific_energy')
            assert out.shape == active_zone_shape
            return out


def add_extra_potential_fields(potential_field_func = _potential_field,
                               potential_field_name = ('cholla', 'gravitational_potential'),
                               potential_field_units = 'code_specific_energy',
                               *, ds = None, skip_additional = False,
                               name_suffix = '', force_override = False):
    """
    This function defines a derived yt-field about the gravitational potential 
    and can optionally define a number of additional derived fields

    Parameters
    ----------
    potential_field_func : callable
        A function that will be passed on to yt.add_field orfield, that
        actually defines the gravitational potential.
    potential_field_name : tuple of 2 strs
        Specifies the field name of the gravitational potential.
    potential_field_units : str
        the units of the gravitational potential
    ds
        When this is None, derived fields are defined with yt.add_field. Otherwise,
        we use ds.add_field
    skip_additional : bool, optional
        When True, only define the main field
    name_suffix : str, Optional
        An optional suffix appended to the names of additional derived fields
    """
    add_field = yt.add_field
    if ds is not None:
        add_field = ds.add_field

    def _dPhidR(field, data, *, variant = 'pure-deriv'):
        cell_width = data.ds.domain_width / data.ds.domain_dimensions

        potential = potential_field_func(field, data)

        x, y = data['index', 'x'], data['index', 'y']

        dPhi_dx = deriv.grid_deriv(potential, dx = cell_width[0],
                                   nominal_order = 2, axis = 0)
        dPhi_dy = deriv.grid_deriv(potential, dx = cell_width[1],
                                   nominal_order = 2, axis = 1)
        vrot2 = x * dPhi_dx + y * dPhi_dy

        if variant == 'vrot2_from_potential':
            return vrot2.to('km**2 / s**2')
        elif variant == 'vrot_from_potential':
            return np.sqrt(np.abs(vrot2)).to('km / s')
        elif variant == 'pure-deriv':
            rcyl = np.sqrt(x * x + y * y)
            return (vrot2 / rcyl).to('code_specific_energy/code_length')
        else:
            rcyl = np.sqrt(x * x + y * y)
            return rcyl, (vrot2 / rcyl)

    def radial_support(field, data):
        cell_width = data.ds.domain_width / data.ds.domain_dimensions
        inverse_density = 1.0/data['cholla','density']
        pressure = data['gas', 'pressure']
        x, y = data['index', 'x'], data['index', 'y']
        rcyl = np.sqrt(x * x + y * y)
        dP_dx = deriv.grid_deriv(pressure, dx = cell_width[0],
                                 nominal_order = 2, axis = 0)
        dP_dy = deriv.grid_deriv(pressure, dx = cell_width[1],
                                 nominal_order = 2, axis = 1)
        dP_dr = (x * dP_dx + y * dP_dy) / rcyl
        invrho_times_dP_div_dR = dP_dr * inverse_density

        vrot2_div_R = (
            np.square(inverse_density * _momentum_cylindrical_phi(field, data))
            / rcyl)

        return (vrot2_div_R - invrho_times_dP_div_dR).to('code_length/code_time**2')

    def radial_hse_term(field, data):
        
        dPhi_div_dR = _dPhidR(field, data, variant = 'pure-deriv')

        # vrot2_div_R is the acceleration from centrifugal force
        # dPhi_div_dR is the acceleration from gravity
        # invrho_times_dP_div_dR is acceleration from pressure gradient

        unnormalized = radial_support(field, data) - dPhi_div_dR
        return (unnormalized/np.abs(dPhi_div_dR)).to('dimensionless')

    def _dPhidz(field, data):
        cell_width = data.ds.domain_width / data.ds.domain_dimensions
        potential = potential_field_func(field, data)
        dPhi_dz = deriv.grid_deriv(potential, dx = cell_width[2],
                                   nominal_order = 2, axis = 2)
        return dPhi_dz.to('code_specific_energy/code_length')

    def vertical_hse(field, data):
        cell_width = data.ds.domain_width / data.ds.domain_dimensions
        pressure = data['gas', 'pressure']
        dP_dz = deriv.grid_deriv(pressure, dx = cell_width[2],
                                 nominal_order = 2, axis = 0)
        inverse_density = 1.0/data['cholla','density']
        invrho_times_dP_div_dz = dP_dz * inverse_density

        dPhi_dz = _dPhidz(field, data)
        return (np.abs(invrho_times_dP_div_dz) - np.abs(dPhi_dz)) / np.abs(dPhi_dz)

    suf = name_suffix

    kwargs_l = [
        {'function' : potential_field_func, 
         'name' : potential_field_name,
         'units' : potential_field_units,
        },

        {'function' : partial(_dPhidR, variant = 'pure-deriv'),
         'name' : ("cholla", f"gravitational_potential_gradient_R{suf}"),
         'units' : 'code_specific_energy/code_length',
        },

        {'function' : partial(_dPhidR, variant = 'vrot2_from_potential'),
         'name' : ("cholla", f"vrot2_from_potential{suf}"),
         'units' : '(km/s)**2',
        },

        {'function' : partial(_dPhidR, variant = 'vrot_from_potential'),
         'name' : ("cholla", f"vrot_from_potential{suf}"),
         'units' : 'km/s',
        },

        {'function' : radial_support,
         'name' : ("cholla", f"radial_accel_support{suf}"),
         'units' : 'code_length/code_time**2',
         'display_name' : (
             r"$\left(\frac{v_{\rm rot}^2}{R} -  "
             r"\frac{1}{\rho}\ \frac{\partial P}{\partial R}"
             r"\right)$"),
        },

        {'function' : radial_hse_term,
         'name' : ("cholla", f"hse{suf}"),
         'units' : 'dimensionless',
         'display_name' : (
            r"$\left(\frac{v_{\rm rot}^2}{R} -  "
            r"\frac{1}{\rho}\ \frac{\partial P}{\partial R} + "
            r"\frac{\partial \Phi}{\partial R}\right)\ / "
            r"\left|\frac{\partial \Phi}{\partial R}\right|$"),
        },

        {'function' : _dPhidz,
         'name' : ("cholla", f"gravitational_potential_gradient_z{suf}"),
         'units' : 'code_specific_energy/code_length',
        },

        {'function' : vertical_hse,
         'name' : ("cholla", f"vertical_hse{suf}"),
         'units' : 'dimensionless',
         'display_name' :
            r"$\left( "
            r"\frac{1}{\rho}\ \frac{\partial P}{\partial z} + "
            r"\frac{\partial \Phi}{\partial z}\right)\ / "
            r"\left|\frac{\partial \Phi}{\partial z}\right|$"),
        },
    ]

    if skip_additional:
        kwargs_l = kwargs_l[:1]

    field_list = []
    for kwargs in kwargs_l:
        add_field(
            sampling_type = 'cell', take_log = False,
            validators = yt.ValidateGridType(),
            force_override = force_override,
            **kwargs)
        field_list.append(kwargs['name'])
    return field_list