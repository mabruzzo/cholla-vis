#!/usr/bin/env python3
"""
A general utility for making profiles from Cholla snapshots
"""

from cholla_vis.profile_creator import (
    IOConfig, create_profile, _list_profile_choices, show_profile_choices
)

import argparse
import os
import yt

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

def main_plot(args):
    yt.enable_parallelism()

    if args.parallel_snap is None:
        parallel = True
    else:
        assert(args.parallel_snap>=1 and isinstance(args.parallel_snap,int))
        parallel = args.parallel_snap

    assert len(set(args.sim)) == len(args.sim)
    
    for sim_name in args.sim:
        create_profile(args.profile_kind, mk_testioconf(sim_name), parallel=parallel)

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

parser = argparse.ArgumentParser(
    prog='make_profile', description= __doc__
)
subparsers = parser.add_subparsers(required=True)
prof_parser = subparsers.add_parser(
    "make", help="actually creates the profiles"
)
prof_parser.add_argument(
    "--profile-kind",
    choices=_list_profile_choices(),
    required=True,
    help="use the show-kind subcommand for more details"
)
prof_parser.add_argument(
    "--sim",
    choices=_SIM_CHOICES,
    nargs='+'
)
prof_parser.add_argument(
    "--parallel_snap", action = 'store', default = None, type = int,
    help=(
        "the number of snapshots that should be processed in parallel. A "
        "value of 1 means that only 1 is processed at a time. The remaining "
        "processors work together to process the data. The total number of "
        "processors must be evenly divisible by this value."
    )
)
prof_parser.set_defaults(func=main_plot)

def main_showkind(args):
    show_profile_choices()

showkind_parser = subparsers.add_parser(
    "show-kind", help="lists and describes the available --profile-kind options"
)
showkind_parser.set_defaults(func=main_showkind)


if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)