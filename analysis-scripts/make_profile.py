#!/usr/bin/env python3
"""
A general utility for making profiles from Cholla snapshots
"""

from cholla_vis.profile_creator import (
    IOConfig, create_profile, _list_profile_choices, show_profile_choices
)
from cholla_vis.conf import (
    PathConf, full_conf_from_cli, build_conf_parser
)
from cholla_vis.parallel_spec import (
    build_parallel_spec_parser, parallel_spec_from_cli,
    check_and_summarize_parallel_spec
)

import argparse
from collections.abc import Sequence
import os
import re
import yt

def _integer_sequence(s: str):
    # This is taken from the scripts in the cholla codebase
    #
    # converts an argument string to an integer sequence
    # -> s can be a range specified as start:stop:step. This follows mirrors
    #    the semantics of a python slice (at the moment, start and stop are
    #    both required)
    # -> s can b a comma separated list
    # -> s can be a single value
    m = re.match(r"(?P<start>[-+]?\d+):(?P<stop>[-+]?\d+)(:(?P<step>[-+]?\d+))?", s)
    if m is not None:
        rslts = m.groupdict()
        step = 1
        if rslts["step"] is not None:
            step = int(rslts.get("step", 1))
        if step == 0:
            raise ValueError(f"The range, {s!r}, has a stepsize of 0")
        seq = range(int(rslts["start"]), int(rslts["stop"]), step)
        if len(seq) == 0:
            raise ValueError(f"The range, {s!r}, has 0 values")
        return seq
    elif re.match(r"([-+]?\d+)(,[ ]*[-+]?\d+)+", s):
        seq = [int(elem) for elem in s.split(",")]
        return seq
    try:
        return [int(s)]
    except ValueError:
        raise ValueError(
            f"{s!r} is invalid. It should be a single int or a range"
        ) from None
    
def main_plot(args):

    yt.enable_parallelism()
    parallel_spec = parallel_spec_from_cli(args)
    check_and_summarize_parallel_spec(parallel_spec)

    assert len(set(args.sim)) == len(args.sim)

    full_conf = full_conf_from_cli(args=args)
    simprop_conf = full_conf.simprop_conf

    # perform a sanity-check that all simulations are valid
    name_snap_pairs = []
    for sim_arg in args.sim:
        if "," in sim_arg:
            sim_name, integer_seq_str = sim_arg.split(",")
        else:
            sim_name, integer_seq_str = sim_arg, None
        sim_props = simprop_conf.data.get(sim_name, None)
        if sim_props is None:
            raise RuntimeError(
                f"{sim_name!r} is not a know simulation. Known names include: " +
                ", ".join(simprop_conf.data.keys())
            )
        elif integer_seq_str is None:
            name_snap_pairs.append(
                (sim_name, range(sim_props.start_snap, sim_props.stop_snap))
            )
        else:
            name_snap_pairs.append((sim_name, _integer_sequence(integer_seq_str)))

    comm = yt.communication_system.communicators[-1]
    for name, snap_seq in name_snap_pairs:
        if comm.rank == 0:
            print(f"begin processing: sim = {name!r}, snaps = {snap_seq}", flush=True)
        comm.barrier()
        ioconf = mk_ioconf(
            sim_name=sim_name, path_conf=full_conf.path_conf, snaps=snap_seq
        )
        create_profile(args.profile_kind, ioconf=ioconf, parallel_spec=parallel_spec)

def mk_ioconf(sim_name: str, path_conf: PathConf, snaps: Sequence[int]) -> IOConfig:
    simdir = path_conf.simdata_prefix
    datadir = path_conf.intermediate_data
    fname_template = os.path.join(simdir, sim_name, 'cat', "{snap:d}.h5")

    return IOConfig(
        snap_seq = snaps,
        fname_template=fname_template,
        outdir=os.path.join(datadir, sim_name)
    )

parser = argparse.ArgumentParser(
    prog='make_profile', description= __doc__
)
subparsers = parser.add_subparsers(required=True)
prof_parser = subparsers.add_parser(
    "make",
    help="actually creates the profiles",
    parents=[build_conf_parser(), build_parallel_spec_parser()]
)
prof_parser.add_argument(
    "--profile-kind",
    choices=_list_profile_choices(),
    required=True,
    help="use the show-kind subcommand for more details"
)
prof_parser.add_argument(
    "--sim",
    nargs='+',
    required=True,
    help=(
        "specify the names of the simulations to make profiles for. You can optionally "
        "specify a snapshot range by passing '<sim-name>,<snap-range>' instead of "
        "just <sim-name>. For example, '<sim-name>,850:871:10' would select snapshots "
        "850, 860, and 870"
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