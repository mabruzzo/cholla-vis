# we need to unify this with galaxy_profile

import argparse
from dataclasses import dataclass
import os

from functools import wraps,partial
from time import time

import numpy as np
import unyt
import yt

if __name__ == '__main__':
    yt.enable_parallelism()

from cholla_vis import galaxy_profile

_TBIN_EDGES = [
    (0.0, None),
    (5050.0, r'5050 {\rm K}'),
    (2e4, r'2\times 10^4 {\rm K}'),
    (5e5, r'5\times 10^5 {\rm K}'),
    (np.finfo(np.float32).max, None),
]

def _standard_T_fieldbin_pair():
    bins = [v for v,_ in _TBIN_EDGES]
    return ("gas","temperature"), unyt.unyt_array(bins, 'K')

def _standard_T_binnames():
    _, bins = _standard_T_fieldbin_pair()
    names = []
    for i in range(bins.size-1):
        left  = '' if i == 0 else rf'{_TBIN_EDGES[i][1]} \leq '
        right = '' if i == bins.size-2 else rf'< {_TBIN_EDGES[i+1][1]}'
        names.append('$' + left + 'T' + right + '$')
    return names

@dataclass(frozen=True)
class AnnuliSpec:
    nominal_radius: float
    length_units: str
    num_in_nominal_radius: int

    def __post_init__(self):
        assert self.nominal_radius > 0
        assert self.num_in_nominal_radius > 0

    def annulus_area_div_pi(self):
        return (
            (self.nominal_radius * self.nominal_radius) /
            self.num_in_nominal_radius
        )

    def radii_thru_max_radius_squared(self, max_radius_sq):
        annulus_area_div_pi = self.annulus_area_div_pi()
        # max_radius_sq times pi gives area of the circle with radius
        # sqrt(max_radius_sq). Thus, the number of complete annuli that
        # at least partially overlap with this circle is given by the
        # following (we add 1 for safety with floats)
        n_annuli = int(np.ceil(max_radius_sq / annulus_area_div_pi))+1

        cumulative_areas_div_pi = (
            np.arange(0, n_annuli + 1) * annulus_area_div_pi
        )

        radii = np.sqrt(cumulative_areas_div_pi)
        if n_annuli >= self.num_in_nominal_radius:
            radii[self.num_in_nominal_radius] = self.nominal_radius
        return radii

    def radii_for_rectangle(self, widths):
        """
        Specifies the radii bounding each concentric annulus, centered
        on the specified rectangle that at least partially overlaps with the
        rectangle
        """
        assert len(widths) == 2
        # distance from origin to the corner
        max_radius_sq = 0.25*((widths[0]*widths[0]) + (widths[1]*widths[1]) )
        return self.radii_thru_max_radius_squared(max_radius_sq)

def _standard_annuli_spec(use_code_length=True):
    length_units = "code_length" if use_code_length else "kpc"
    return AnnuliSpec(
        nominal_radius = 1.2, length_units = length_units, num_in_nominal_radius = 12
    )


def _get_bin_kwargs(ds, use_sph_radial):
    max_z = np.maximum(
        np.abs(ds.domain_left_edge[2]),
        np.abs(ds.domain_right_edge[2])
    ).to('code_length').v

    max_dist = max_z

    if use_sph_radial:
        dist_field = ('index', 'spherical_radius')
    else:
        dist_field = galaxy_profile._ABSZ_FIELD

    distbinwidth = ds.quan(0.125, "code_length")
    n_distbins = int(np.ceil(max_dist / distbinwidth.v))
    distbins = distbinwidth * np.arange(n_distbins+1)

    if use_sph_radial:
        ybin_field_name = ("index", "symmetric_theta")
        ybins = ds.arr(
            [0.0, np.pi/12.0, np.pi/6.0, np.pi/4.0, np.pi/3.0, 5*np.pi/12.0, np.pi/2.0],
            'radian'
        )
    else:
        ybin_field_name = ("index", "cylindrical_radius")
        ann_spec = _standard_annuli_spec(use_code_length=True)
        ybins = ds.arr(
            ann_spec.radii_for_rectangle(ds.domain_width[:2].to('code_length').v),
            'code_length'
        )
    override_bins_with_unit = {
        dist_field : distbins,
        ybin_field_name : ybins,
        ("gas", "temperature") : _standard_T_fieldbin_pair()[1]
    }

    kwargs = {
        'bin_fields' : list(override_bins_with_unit.keys()),
        'override_bins' : {
            k : v.ndview for k,v in override_bins_with_unit.items()
        },
        'units' : {
            k : str(v.units) for k,v in override_bins_with_unit.items()
        }
    }

    return kwargs

def fetch_props_scale_height(ds, classic_bins = True):
    if classic_bins:
        kwargs = galaxy_profile._create_galaxy_profile_binkwargs(
            ds, bins_rcyl = 170, bins_z = 177,
            range_rcyl = [0,1.7], range_z = [0.0,1.77],
            use_abs_z = True,
            extra_field_bins_pair = _standard_T_fieldbin_pair()
        )
    else:
         kwargs = _get_bin_kwargs(ds, use_sph_radial=False)


    return yt.create_profile(
        ds.all_data(), fields = [
            ('gas', 'cell_mass'),
            ("gas", "mass_zsq"),
            ('gas', 'velocity_cylindrical_theta'),
            ('index', 'volume'),
            ("gas", "massT"),
            ("gas", "pressure_tot_slab"),
        ],
        weight_field = None,
        **kwargs
    )


def assorted_binned_properties(ds):

    kwargs = _get_bin_kwargs(ds, use_sph_radial=False)

    return yt.create_profile(
        ds.all_data(), fields = [
            ('gas', 'number_density'),
            ('gas', 'pressure'),
            ('gas', 'temperature'),
            ('gas', 'velocity_spherical_radius'),
            ("gas", "outwardz_velocity"),
            ("gas", "adiabat")
        ],
        weight_field = ("gas", "density"),
        **kwargs
    )

def flux_binned_props(ds, radial, only_positive_vel = False):
    kwargs = _get_bin_kwargs(ds, use_sph_radial=radial)

    if radial:
        fields = [
            ("gas", "flux_rho_spherical_radius"),
            ("gas", "flux_momentum_spherical_radius"),
            ("gas", "flux_e_spherical_radius")
        ]
        positive_vel_field = (
            "index", "has_positive_velocity_spherical_radius"
        )
    else:
        fields = [
            ("gas", "flux_rho_outwardz"),
            ("gas", "flux_momentum_outwardz"),
            ("gas", "flux_e_outwardz")
        ]
        positive_vel_field = (
            "index", "has_positive_outwardz_velocity"
        )

    if only_positive_vel:
        weight_field = positive_vel_field
    else:
        weight_field = ("index", "ones")

    return yt.create_profile(
        ds.all_data(), fields = fields,
        weight_field = weight_field,
        **kwargs
    )

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap


def create_series(series, callback, fname_func):
    """
    Generates yt profiles (or anything else)

    Parameters
    ----------
    series: DatasetSeries
        The series of datasets to load the files from
    callback: callable
        Operates 
    fname_func: callable, optional
        A callable that when given a dataset will yield the filename to save 
        the profile to
    """

    my_storage = {}
    for sto,ds in series.piter(storage = my_storage):
        print(str(ds))
        if fname_func is not None:
            fname = fname_func(ds)
        else:
            fname = None

        callback(ds, fname)

        #if resume_manager is None:
        #    create_profile(ds = ds, out_fname = fname)
        #else:
        #    resume_manager.process_ds(ds, fname, create_profile)
        sto.result = fname
        # the following may be necessary help with memory
        ds.index.clear_all_data()

    return [fname for i, fname in sorted(my_storage.items())]

@dataclass
class IOConfig:
    fnames: list[str]
    outdir: str

_FLUX_PROFILES = ('r_fluxes', 'r_fluxes_positive',
                  'z_fluxes', 'z_fluxes_positive')

_CHOICES = _FLUX_PROFILES + ('scale-height', 'assorted')

def main(profile_kind, ioconf, parallel=True):

    def setup(ds): # do nothing
        galaxy_profile.try_add_absz(ds)
        galaxy_profile.try_add_extra_fields(ds)
        galaxy_profile.add_flux_fields(ds, radial=True)
        galaxy_profile.add_flux_fields(ds, radial=False)
    print(f"parallel = {parallel}")

    series = yt.DatasetSeries(ioconf.fnames,
                              parallel = parallel,
                              setup_function = setup)
    if profile_kind in _FLUX_PROFILES:
        fn = partial(
            flux_binned_props,
            radial=profile_kind.startswith('r_'),
            only_positive_vel=profile_kind.endswith('_positive')
        )
    elif profile_kind == 'scale-height':
        fn = fetch_props_scale_height
    else:
        fn = assorted_binned_properties

    dtypename = profile_kind
    dirname = os.path.join(ioconf.outdir, dtypename)
    def fname_func(ds):
        s = str(ds)
        if s.endswith('.h5.0'):
            basename = s[:-2]
        else:
            basename = s
        return os.path.join(dirname, basename)

    os.makedirs(dirname, exist_ok=True)

    @timing
    def do_work(ds, fname):
        prof = fn(ds)
        prof.save_as_dataset(fname)
        return prof

    fnames = create_series(series, do_work, fname_func)
    print("function is complete! Created: ")
    for fname in fnames:
        print("-> ", fname)

_STEP = 10
_CASE_DICT = {
    '708cube_GasStaticG-1Einj_restart-TIcool' : range(850, 1485, _STEP),
    #'708cube_GasStaticG-1Einj_restart-TIcool' : range(1000, 1485, 1),
    '708cube_GasStaticG-1Einj' : range(800, 1334, _STEP),
    '708cube_GasStaticG-1Einj_restartDelay-TIcool' : range(1300, 1461, _STEP),
    '708cube_GasStaticG-2Einj_restart-TIcool' : range(1205, 1591, _STEP),
    '708cube_GasStaticG-2Einj' : range(850, 1081, _STEP),
}

_SIM_CHOICES = list(_CASE_DICT.keys())

def mk_testioconf(sim_name):
    _SIMDIR = "/ix/eschneider/mabruzzo/hydro/galactic-center/"
    _DATADIR = "/ix/eschneider/mabruzzo/hydro/galactic-center/analysis-data/"

    snaps = _CASE_DICT[sim_name]

    fnames = [
        os.path.join(_SIMDIR, sim_name, 'raw', f"{snap}", f"{snap}.h5.0")
        for snap in snaps
    ]
    return IOConfig(
        fnames=fnames,
        outdir=os.path.join(_DATADIR, sim_name)
    )

parser = argparse.ArgumentParser(prog='make_profile')
parser.add_argument("--profile-kind", choices=_CHOICES)
parser.add_argument("--sim", choices=_SIM_CHOICES, nargs='+')
parser.add_argument(
    "--parallel_snap", action = 'store', default = None, type = int,
    help=(
        "the number of snapshots that should be processed in parallel. A "
        "value of 1 means that only 1 is processed at a time. The remaining "
        "processors work together to process the data. The total number of "
        "processors must be evenly divisible by this value."
    )
)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.parallel_snap is None:
        parallel = True
    else:
        assert(args.parallel_snap>=1 and isinstance(args.parallel_snap,int))
        parallel = args.parallel_snap

    assert len(set(args.sim)) == len(args.sim)
    
    for sim_name in args.sim:
        main(args.profile_kind, mk_testioconf(sim_name), parallel=parallel)
