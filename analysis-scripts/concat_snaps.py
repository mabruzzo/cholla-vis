"""
A tool to help with concatenation.

In principle, it supports the MPI Executor. It's been a while since I enabled that (&
it requires manual intervention).

At this point, I think this is fairly redundant with some of the existing python-scripts
"""

USE_MPI_EXECUTOR = False

if USE_MPI_EXECUTOR:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    from mpi4py.futures import MPIPoolExecutor
else:
    rank = 0

import argparse
import datetime
import glob
import functools
import importlib.util
import os
import pathlib
import re
import sys
from typing import Callable, Optional

# a global cache variable for _get_concat_fn
_LOADED_CONCAT_FN: Optional[Callable] = None


def _get_concat_fn(cholla_src_dir: str) -> Callable:
    """
    A hacky way to get the concatenation function, which is defined in the cholla
    directory

    the implementation is inspired by
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    global _LOADED_CONCAT_FN  # Declares '_LOADED_CONCAT_FN' as a global variable
    module_names = ["concat_internals", "concat_2d_data"]
    rslt_is_cached = _LOADED_CONCAT_FN is not None
    if not rslt_is_cached and any(m in sys.modules for m in module_names):
        raise RuntimeError(
            f"Unexpected scenario: a module with one of the names, {module_names} "
            "was already loaded"
        )
    elif not rslt_is_cached:
        l = []
        for m in module_names:
            file_path = os.path.abspath(
                os.path.join(cholla_src_dir, "python_scripts", f"{m}.py")
            )
            spec = importlib.util.spec_from_file_location(m, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[m] = module
            spec.loader.exec_module(module)
            l.append(module)
        _LOADED_CONCAT_FN = l[-1].concat_2d_dataset
    return _LOADED_CONCAT_FN


def full_execute(
    output_number: int,
    fn_map: Callable,
    num_processes: int,
    output_dir: str,
    cholla_src_dir: str,
    cleanup: bool = False,
):
    """
    Actually performs the work of concatenating files from a single snapshot
    """

    # get the function to use for concatenation
    concat_2d_dataset = _get_concat_fn(cholla_src_dir)

    resulting_paths = []

    kinds = ["proj", "slice"]

    for kind in kinds:
        resulting_paths.append(f"{output_dir}/{output_number}_{kind}.h5")

    already_exists = all(os.path.isfile(path) for path in resulting_paths)

    timings = {}
    if not already_exists:
        for kind in kinds:
            t1 = datetime.datetime.now()
            concat_2d_dataset(
                output_directory=pathlib.Path(output_dir),
                num_processes=num_processes,
                output_number=output_number,
                dataset_kind=kind,
                build_source_path=fn_map[kind],
                concat_xy=True,
                concat_yz=True,
                concat_xz=True,
            )
            t2 = datetime.datetime.now()
            timings[kind] = (t2 - t1).total_seconds()

        if cleanup:
            for kind in ["proj", "slice"]:
                l = glob.glob(fn_map[kind](proc_id="*", nfile=output_number))
                for path in l:
                    os.remove(path)
    return (rank, output_number, timings)


def integer_sequence(s):
    # this was copy-pasted, which is unfortunate!
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


parser = argparse.ArgumentParser(description="Concatenate 2d cholla datasets")
parser.add_argument(
    "--snaps",
    type=integer_sequence,
    required=True,
    help=(
        "Which indices to plot. Can be a single number (e.g. 8) or "
        "a range specified with slice syntax (e.g. 2:9 or 5:3)."
    ),
)
parser.add_argument(
    "-s",
    "--source-directory",
    required=True,
    help="Path to the directory for the source HDF5 files",
)
parser.add_argument(
    "--proc-per-snap",
    type=int,
    required=True,
    help="Number of processes per snapshot used during the simulation",
)
parser.add_argument(
    "--cholla-src-dir", required=True, help="path to the cholla source directory"
)
parser.add_argument(
    "--cleanup",
    action="store_true",
    help="remove the original files after concatenation",
)

_DEFAULT_OUTDIR = "./catfiles/"
parser.add_argument(
    "--output-directory",
    default=_DEFAULT_OUTDIR,
    help=f"specifies where results get written (default: {_DEFAULT_OUTDIR})",
)

if __name__ == "__main__":
    args = parser.parse_args()
    itr = args.snaps

    def build_source_path(proc_id: int, nfile: int, kind: str, dirpath: str) -> str:
        return f"{dirpath}/{nfile}/{nfile}_{kind}.h5.{proc_id}"

    fn_map = {
        "slice": functools.partial(
            build_source_path, kind="slice", dirpath=args.source_directory
        ),
        "proj": functools.partial(
            build_source_path, kind="proj", dirpath=args.source_directory
        ),
    }

    execute = functools.partial(
        full_execute,
        fn_map=fn_map,
        num_processes=args.proc_per_snap,
        output_dir=args.output_directory,
        cholla_src_dir=args.cholla_src_dir,
        cleanup=args.cleanup,
    )

    if USE_MPI_EXECUTOR:
        with MPIPoolExecutor() as executor:
            for rslt in executor.map(execute, itr):
                print(rslt)
    else:
        for rslt in map(execute, itr):
            print(rslt)
