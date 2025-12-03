import argparse
from dataclasses import dataclass
import sys

@dataclass(frozen=True)
class ParallelSpec:
    # number of jobs to spawn. By default, the number of jobs matches the number of
    # MPI ranks. If there are 6 MPI ranks and njobs is 2, then the list of snapshots
    # will be split over 2 groups of 3 ranks
    n_work_groups: int = 0
    # indicates whether there is dynamic load balancing (an MPI rank is dedicated to
    # this task)
    dynamic_balance: bool = False

def build_parallel_spec_parser() -> argparse.ArgumentParser:
    """Builds an argument parser designed for use in scripts

    To use these arguments alongside other arguments, you might initialize this
    parser first and use the parents kwarg when constructing the script's
    parser.
    """
    p = argparse.ArgumentParser(add_help=False)
    grp = p.add_argument_group(
        "parallelism",
        description=(
            "These arguments customizes the MPI parallelism used in the program. "
            "The main work is distributed among working-groups of MPI processes. "
            "Essentially, snapshots are distributed among working-groups. All MPI "
            "processes of a working-group collaboratively/synchronously perform the "
            "full calculation for an assigned snapshot, and once the work is "
            "complete, all the members in the group move onto the next assigned "
            "snapshot. In other words, if there are `N_WORK_GROUPS`, then up to "
            "`N_WORK_GROUPS` different snapshots can be processed at once. All "
            "available MPI ranks are evenly divided among the working-groups (the "
            "program reports an error if they can't be evenly divided). The number of "
            "available MPI processes depends on whether dynamic load balancing is "
            "enabled (it's disabled by default). If the program is launched with "
            "`TOTAL_SIZE` MPI ranks, then the number of available ranks is "
            "`TOTAL_SIZE` without dynamic balancing or `TOTAL_SIZE-1` with dynamic "
            "balancing."
        )
    )
    grp.add_argument(
        "--dynamic-balance", action="store_true", help="enables dynamic load balancing"
    )
    grp.add_argument(
        "--n-work-groups",
        action='store',
        default=None,
        type=int,
        metavar="N_WORK_GROUPS",
        help=(
            "Sets the number of working group. By default, `N_WORK_GROUPS` is equal "
            "to the number of available MPI ranks (i.e. each group is composed of 1 "
            "MPI process)"
        )
    )
    return p

def parallel_spec_from_cli(args: argparse.Namespace) -> ParallelSpec:

    kwargs = {"dynamic_balance": args.dynamic_balance}
    if args.n_work_groups is not None:
        assert(args.n_work_groups>=1 and isinstance(args.n_work_groups,int))
        kwargs["n_work_groups"] = args.n_work_groups
    return ParallelSpec(**kwargs)


def check_and_summarize_parallel_spec(parallel_spec: ParallelSpec):
    # this should be called after yt.enable_parallelism
    import yt
    comm = yt.communication_system.communicators[-1]

    total_size = comm.size
    if (total_size == 1) and parallel_spec.dynamic_balance:
        raise ValueError(
            "Can't enable dynamic load balancing when run with 1 process"
        )
    elif parallel_spec.dynamic_balance:
        available_size = total_size - 1
    else:
        available_size = total_size

    if parallel_spec.n_work_groups == 0:
        n_work_groups = available_size
    else:
        n_work_groups = parallel_spec.n_work_groups

    if comm.rank == 0:
        ranks_per_group = available_size // n_work_groups
        print(
            "Parallelism Summary:\n"
            f"-> Total Number of MPI processes: {total_size}\n"
            f"-> Use dynamic load balancing: {parallel_spec.dynamic_balance}\n"
            f"-> Number of Available MPI processes: {available_size}\n"
            f"-> Number of Work-Groups: {n_work_groups}\n"
            f"-> MPI Processes per Group: {ranks_per_group}"
        )
        sys.stdout.flush()
    comm.barrier()
    
    if (available_size % n_work_groups) != 0:
        raise ValueError(
            f"the number of MPI processes, {available_size}, is not evenly divisible "
            f"by n_work_groups, {n_work_groups}"
        )    
