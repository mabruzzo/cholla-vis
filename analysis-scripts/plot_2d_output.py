#!/usr/bin/env python3
"""
A general utility for making simple plots from projection and slice output data files
written by a Cholla simulation.

While this utility nominally supports working with distributed data files, experience
indicates that the plots need to get remade with some frequency, and so it's usually
beneficial to concatenate the relevant data files ahead of time.
"""

# command line parsing
import argparse
from collections.abc import Sequence
import contextlib
import multiprocessing
import re
from typing import NamedTuple

from cholla_vis.sample_plot import make_plots, _get_known_presets


class DSetKind(NamedTuple):
    plot_proj: bool
    known_orientations: tuple[str, ...]
    default_presets: tuple[str, ...]


_PLOT_PROPS = {
    "proj": DSetKind(
        plot_proj=True,
        known_orientations=("xy", "xz"),
        default_presets=("column_density", "avg_temperature"),
    ),
    "slice": DSetKind(
        plot_proj=False,
        known_orientations=("yz", "xy", "xz"),
        default_presets=("temperature", "ndens", "phat"),
    ),
}


def _coerce_dir_path(flag_name: str, args: argparse.Namespace, fallback: str) -> str:
    attr_name = flag_name.lstrip("-").replace("-", "_")
    assert len(attr_name) > 0  # sanity check!
    attr_val = getattr(args, attr_name)
    if attr_val is None:
        return "./"
    elif len(attr_val) == 0:
        raise ValueError(f"somehow {flag_name} was passed an empty string")
    elif attr_val.endswith("/"):
        return attr_val
    else:
        return attr_val + "/"


def main_plot(args: argparse.Namespace):
    plot_prop = _PLOT_PROPS[args.kind]

    run_dir = _coerce_dir_path("--load-dir", args, fallback="./")
    outdir_prefix = _coerce_dir_path("--save-dir", args, fallback="./")

    snap_list = args.snaps
    preset_name = plot_prop.default_presets if args.quan is None else args.quan

    if args.ncores == 1:
        pool_context_manager = contextlib.nullcontext(enter_result=None)
    elif args.ncores > 1:
        pool_context_manager = multiprocessing.Pool(processes=args.ncores)
    else:
        raise RuntimeError("the --ncores arg must be a positive int")

    with pool_context_manager as pool:
        try:
            make_plots(
                snap_list,
                run_dir=run_dir,
                preset_name=preset_name,
                orientation=plot_prop.known_orientations,
                plot_proj=plot_prop.plot_proj,
                pool=pool,  # <- either None or multiprocessing.Pool
                outdir_prefix=outdir_prefix,
                load_distributed_files=args.distributed_load,
            )
        except Exception as e:
            # I think we historically did something a little more clever here...
            err_message = f"encountered problem"
            print(err_message)
            raise


def main_showpresets(args: argparse.Namespace):
    print("showing presets")
    for kind in ["slice", "proj"]:
        print(f"{kind}:")
        print(f"  {list(_get_known_presets(kind))!r}")


# should probably get factored out
def integer_sequence(s: str) -> Sequence[int]:
    """
    Parse the string `s` as a sequence of integers.
    """

    m = re.match(r"(?P<start>[-+]?\d+):(?P<stop>[-+]?\d+)(:(?P<step>[-+]?\d+))?", s)
    if m is not None:
        rslts = m.groupdict()
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


parser = argparse.ArgumentParser(description=__doc__)
subparsers = parser.add_subparsers(required=True)

plot_parser = subparsers.add_parser("plot")
plot_parser.add_argument(
    "--snaps",
    type=integer_sequence,
    required=True,
    help="Which indices to plot. Can be a single number (e.g. 8) or "
    "a range specified with slice syntax (e.g. 2:9 or 5:3). ",
)
plot_parser.add_argument(
    "--load-dir", type=str, default="./", help="Specifies directory to load data from"
)
plot_parser.add_argument(
    "--save-dir", type=str, default=None, help="Specifies directory to save data in"
)
plot_parser.add_argument(
    "-k",
    "--kind",
    choices=["slice", "proj"],
    required=True,
    help="What kind of 2D dataset to plot",
)
plot_parser.add_argument("--quan", type=str, nargs="+", help="The quantity to plot")
plot_parser.add_argument(
    "--ncores", type=int, default=1, help="Number of processes (using multiprocessing)"
)
plot_parser.add_argument(
    "--distributed-load",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="specifies whether each files are distributed (or concatenated)",
)
plot_parser.set_defaults(func=main_plot)

showpreset_parser = subparsers.add_parser("show-presets")
showpreset_parser.set_defaults(func=main_showpresets)

if __name__ == "__main__":
    args = parser.parse_args()
    fn = args.func
    fn(args)
