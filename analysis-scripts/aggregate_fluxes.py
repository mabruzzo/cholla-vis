"""
A utility for aggregating all of the flux data from previously created files.

For a given simulation, the `make_profile.py` script should be used to create profiles
holding flux data for every snapshot of interest. Essentially, we should be generating
4 different profiles:
  (i) radial-net-fluxes (i.e. the "r_fluxes" intermediate data product)
  (ii) radial-outflowing-fluxes (i.e. the "r_fluxes_positive" intermediate data product) 
  (iii) z-net-fluxes (i.e. the "z_fluxes" intermediate data product)
  (iv) z-outflowing-fluxes (i.e. the "z_fluxes_positive" intermediate data product)

Once you are done creating those profiles, this script can be called to aggrate the
time serie data into 2 output hdf5 files for the simulations. One output file will hold
the time series of radial fluxes. The other output file will hold the time series of
z fluxes.
"""

from cholla_vis.conf import full_conf_from_cli, build_conf_parser
from cholla_vis.flux.flux_process import collect_and_save

import argparse

parser = argparse.ArgumentParser(
    prog='aggregate_fluxes',
    description= __doc__,
    parents = [build_conf_parser()]
)
parser.add_argument(
    "--sim",
    nargs='+',
    required=True,
    help="specify the names of the simulations to aggregate fluxes for"
)

def main(args: argparse.Namespace):
    full_conf = full_conf_from_cli(args=args)
    simprop_conf = full_conf.simprop_conf

    # perform a sanity-check that all simulations are valid
    sim_names = args.sim
    for sim_name in args.sim:
        sim_props = simprop_conf.data.get(sim_name, None)
        if sim_props is None:
            raise RuntimeError(
                f"{sim_name!r} is not a know simulation. Known names include: " +
                ", ".join(simprop_conf.data.keys())
            )
    collect_and_save(sim_names=sim_names, path_conf=full_conf.path_conf)

if __name__ == "__main__":
    main(parser.parse_args())