import argparse
from dataclasses import dataclass
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from cholla_vis.log_parse import *

@dataclass(frozen=True)
class SimSetProps:
    base_sim_name : str
    restart_sim_name : str | None
    restart_sim_kyr : float
    prefix_path : str

def get_windowed_SNe_rate(cycle_time_bounds,
                          num_SNe,
                          t_vals_kyr,
                          target_event_count = 100):
    """
    Calculates the SNe rate calculation itself.

    The instantaneous SNe rate uses as an adaptive time window:
    - the definition is taken from
      https://ui.adsabs.harvard.edu/abs/2020ApJ...900...61K/
    - ``SNR(t0)`` is the supernovae count during ``(t0-tau_window) <= t < t0``
      divided by ``tau_window``.
    - as in the above paper, we adaptively pick ``tau_window`` such that 
      ``target_event_count``supernovae occur during the window.

    Parameters
    ----------
    cycle_time_bounds: np.ndarray 
        An (N,2) array specifying the pairs of simulation-times that bound
        a simulation cycle when 1 or more SNe occured. This must satisfy 2
        invariants: (i) ``cycle_time_bounds[:,0] < cycle_time_bounds[:,1]``
        and (ii) ``cycle_time_bounds[::-1,1] <= cycle_time_bounds[1::,0]``
    num_SNe: np.ndarray
        An (N,) array. ``num_SNe[i]`` specifies the (non-zero) number of SNe
        that occured during the timestep starting at ``cycle_time_bounds[i,0]``
        and ending at ``cycle_time_bounds[i,1]``.
    t_vals_kyr: np.ndarray
        A sorted (M,) array that the times (in units of kyr) to compute SNe at
    target_event_count
        A positive integer specifying the target number of SNe to use for
        sizing the window.
    Returns
    -------
    out: pd.DataFrame
        DataFrame holding the computing SNR rate at each time specified by
        ``t_vals_kyr``. Since the exact time of each SNR isn't known to this
        function, the former specifies the lower bound while the latter
        specifies an upper bound (for a sufficiently large
        ``target_event_count``, the uncertainty is small). Rates have negative
        values wherever ``t_vals_kyr < cycle_time_bounds.min()`` or
        ``t_vals_kyr > cycle_time_bounds.max()``.
    """

    out_lo = np.full(fill_value=-1.0, shape = t_vals_kyr.shape)
    out_hi = np.full(fill_value=-1.0, shape = t_vals_kyr.shape)

    # the following logic is structured in a manner similar to the way that we
    # might compute a standard windowed statistic (e.g. mean, stddev) in a
    # single pass
    window_start_idx = 0
    window_end_idx = 0
    window_sum = num_SNe[window_start_idx]

    first_t_val_index = 0
    while t_vals_kyr[first_t_val_index] < cycle_time_bounds[window_end_idx][1]:
        first_t_val_index += 1

    out_lo[:first_t_val_index] = np.nan
    out_hi[:first_t_val_index] = np.nan

    for t_val_index in range(first_t_val_index, len(t_vals_kyr)):
        cur_t_val = t_vals_kyr[t_val_index]
        # advance the right edge of the window as far to the right as possible
        while (
            (window_end_idx+1 < len(cycle_time_bounds)) and
            cycle_time_bounds[window_end_idx+1][1] < cur_t_val
        ):
            window_end_idx+=1
            window_sum += num_SNe[window_end_idx]

        # advance the left edge of the window as far right as possible
        while (window_start_idx+1) <= window_end_idx:
            tmp = window_sum - num_SNe[window_start_idx]
            if tmp < target_event_count:
                break
            window_sum = tmp
            window_start_idx+=1
        longer_diff = cur_t_val - cycle_time_bounds[window_start_idx+1][0]
        shorter_diff = cur_t_val - cycle_time_bounds[window_start_idx+1][1]

        out_lo[t_val_index] = window_sum / longer_diff
        out_hi[t_val_index] = window_sum / shorter_diff

    print("average num per kyr bounds: ", np.nanmean(out_lo), np.nanmean(out_hi))

    return pd.DataFrame(
        data={'num_per_kyr_lo' : out_lo, 'num_per_kyr_hi' : out_hi},
        index=pd.Index(t_vals_kyr, name='t_kyr')
    )

from cholla_vis.log_parse import StepSummaryInfo
import copy

class SNeHistoryBuilder:
    """
    Instances of this type are intended to help calculate the SNe History.

    The basic premise:
    - instances are used as a callback function that is passed into
      ``gather_summary`` to record/aggregate the relevant SNe History
      information that we care about from a log (for computing the SNe rate)
    - the idea is that you might pass in the same instance as you read in logs
    - instances provide extra machinery to create deep copies that only include
      a subset of information
      - we need something like this to make it easy to deal with the
        overlapping times covered in a 2 consecutive logs (i.e. the 2nd log is
        produced by a cholla-execution is produced by restarting from an output
        produced by the cholla-execution that produced the 1st log)
      - our choice to make deepcopies was motivated by the fact that our
        3-phase ISM simulations are restarted from 2-phase ISM simulations.

    Notes
    -----
    The design is horribly over-complicated.
    
    It was originally designed to extract information logged for every single
    SNe event. This includes information about the SNe kind, the particle id,
    the particle's local position. Given information about the particle, I had
    intentions to convert the local position to a domain position (since the
    local blocks are are large & we could fetch the particle's position from
    snapshots).

    However, once I realized that there was a bunch of non-atomic printing, I
    decided to do something far more simple.
    """

    def __init__(self, starting_time=0.0, starting_cycle=0):
        # add one to the cycle_index to get the number of complete iterations
        # at the end of the cycle
        self.cycle_index = []
        self.cycle_time_bounds = []
        self.num_SNe = []
        self._starting_time = starting_time
        self._starting_cycle = starting_cycle

    def set_starting_cycle_and_time(self, starting_cycle, starting_time):
        assert starting_cycle is not None
        assert starting_time is not None
        if self._starting_time is not None:
            assert starting_cycle >= self._starting_cycle
            assert starting_time >= self._starting_time
        if len(self.cycle_time_bounds) > 0:
            assert starting_cycle >= self.cycle_index[-1]
            assert starting_time >= self.cycle_time_bounds[-1][1]
        self._starting_time = starting_time
        self._starting_cycle = starting_cycle

    def make_copy_of_first_nkyr(self, nkyr):
        # todo: use builtin bisect
        length = len(self.cycle_time_bounds)
        assert length > 0
        i = 0
        while (i < length) and self.cycle_time_bounds[i][1] <= nkyr:
            i+=1
        out = copy.copy(self)
        for attr in ['cycle_index', 'cycle_time_bounds', 'num_SNe']:
            l = getattr(self, attr)
            setattr(out, attr, copy.deepcopy(l[:i]))
        return out

    def __call__(
        self,
        prev_step_props: StepSummaryInfo | None,
        cur_step_props: StepSummaryInfo,
        resolved_sn_count: int,
        cur_unresolved_sn_count: int
    ):
        assert (resolved_sn_count + cur_unresolved_sn_count) > 0
        if prev_step_props is None:
            assert self._starting_cycle is not None
            assert self._starting_time is not None
            cur_cycle_ind = self._starting_cycle
            min_cycle_time = self._starting_time
        else:
            cur_cycle_ind = prev_step_props.n_step
            min_cycle_time = prev_step_props.sim_time
        self._starting_cycle, self._starting_time = None, None

        assert cur_cycle_ind + 1 == cur_step_props.n_step
        assert min_cycle_time < cur_step_props.sim_time
        end_cycle_time = cur_step_props.sim_time

        self.cycle_index.append(cur_cycle_ind)
        self.cycle_time_bounds.append((min_cycle_time, end_cycle_time))
        self.num_SNe.append(
            resolved_sn_count + cur_unresolved_sn_count
        )

def concatenated_sne_history(path_l, sne_history_builder = None):
    # this tries to construct
    triples = []
    for path in path_l:
        previouscompletedsteps, starttime \
            = get_previouscompletedsteps_and_starttime(path)
        if len(triples) > 0:
            assert previouscompletedsteps > triples[-1][0]
            assert starttime > triples[-1][1]
        triples.append((previouscompletedsteps, starttime, path))

    if sne_history_builder is None:
        sne_history_builder = SNeHistoryBuilder()

    for i, (first_cycleind, starttime, path) in enumerate(triples):        
        sne_history_builder.set_starting_cycle_and_time(
            first_cycleind, starttime
        )
        if (i+1) < len(triples):
            stop_cycle_index = triples[i+1][0]
        else:
            stop_cycle_index = None

        with open(path, 'r') as f:
            try:
                gather_summary(
                    f,
                    stop_cycle_index = stop_cycle_index,
                    sne_history_builder = sne_history_builder
                )
            except:
                raise RuntimeError(f"problem in {path}")
    return sne_history_builder


def find_SNe_Rates(sim_set_props, target_events_per_window=100):
    """
    Computes the SNe Rates for each simulation specified in a ``SimSetProps``
    instance. Recall that the a ``SimSetProps`` instance specifies a 3-phase
    ISM sim and information about the 2-phase ISM sim that it was restarted
    from.

    In more detail:
    - the ``concatenated_sne_history`` function stitches together the logs
      for a single simulation (it will create a new history-record or extend
      an existing, provided record).
    - get_windowed_SNe_rate does the actual calculation
    """
    l = [(None, sim_set_props.base_sim_name)]
    if sim_set_props.restart_sim_name is not None:
        l.append(
            (sim_set_props.restart_sim_kyr, sim_set_props.restart_sim_name)
        )

    base_builder = None

    out = {}
    for index, (restart_sim_kyr, sim_name) in enumerate(l):
        if index > 0:
            print("\n\n")
        print(f"log-search for {sim_name}")
        glob_pattern = f'{sim_set_props.prefix_path}/{sim_name}/logs-cleaned/slurm-[0-9]*.out'
        log_paths = sorted(glob.glob(glob_pattern))

        print("-> recording SNe History")
        if base_builder is None:
            builder = SNeHistoryBuilder()
            base_builder = builder
        else:
            builder = base_builder.make_copy_of_first_nkyr(restart_sim_kyr)

        concatenated_sne_history(
            log_paths, sne_history_builder = builder
        )
        t_vals_kyr = np.arange(0, 2000) * 100
        t_vals_kyr = t_vals_kyr[t_vals_kyr < builder.cycle_time_bounds[-1][1]]

        df = get_windowed_SNe_rate(
            builder.cycle_time_bounds,
            builder.num_SNe,
            t_vals_kyr=t_vals_kyr,
            target_event_count=target_events_per_window
        )

        if restart_sim_kyr is not None:
            df = df[df.index >= restart_sim_kyr]

        out[sim_name] = df
    return out


def _test_plot(sne_rate_registry):
    import matplotlib.pyplot as plt

    fig,ax_arr = plt.subplots(
        3,2, sharex=True, sharey=True, figsize = (8,8)
    )

    def f(ax_l, pairs):
        for i, (sim_a, sim_b) in enumerate(pairs):
            ax_l[i].plot(sne_rate_registry[sim_a].index/1000,
                    sne_rate_registry[sim_a]['num_per_kyr_hi'],
                    f'C0:',
                    label = "2-phase")
            if sim_b is not None:
                ax_l[i].plot(sne_rate_registry[sim_b].index/1000,
                        sne_rate_registry[sim_b]['num_per_kyr_hi'],
                        f'C0-',
                        label = "3-phase")
    
    pairs_left = [
        ('708cube_GasStaticG-1Einj', None),
        ('708cube_GasStaticG-1Einj',
         '708cube_GasStaticG-1Einj_restart-TIcool'),
        ('708cube_GasStaticG-1Einj',
         '708cube_GasStaticG-1Einj_restartDelay-TIcool'),
    ]
    f(ax_arr[:, 0], pairs_left)
    pairs_right = [
        ('708cube_GasStaticG-2Einj', None),
        ('708cube_GasStaticG-2Einj',
         '708cube_GasStaticG-2Einj_restart-TIcool'),
    ]
    f(ax_arr[:, 1], pairs_right)

    row_labels = [
        "2-phase", "3-phase (after 80 Myr)", "3-phase (after 130 Myr)"
    ]

    for i, row_label in enumerate(row_labels):
        ax_arr[i, 0].set_ylabel(
            row_label + "\n" + r"SN Rate [${\rm kyr}^{-1}$]"
        )

    for i in range(2):
        ax_arr[-1, i].set_xlabel(r"$t$ [Myr]")

    ax_arr[0, 0].annotate(
        r"$E_{\rm inj} = 10^{51}\, {\rm erg}$",
        xy=(0.5, 1), xytext=(0, 5), xycoords='axes fraction',
        textcoords='offset points', size='large', ha='center',
        va='baseline'
    )
    ax_arr[0, 1].annotate(
        r"$E_{\rm inj} = 2\times 10^{51}\, {\rm erg}$",
        xy=(0.5, 1), xytext=(0, 5),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline'
    )

    ax_arr[0,0].set_ylim(0.5, 1.5)

    fig.tight_layout()
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sims-dir",
    required=True,
    help="path to the directory holding simulation data"
)
parser.add_argument(
    "--show-SNe-history-plot",
    action="store_true",
    help="when specified, the SNe history will be plotted"
)
parser.add_argument(
    "--out-dir",
    default=None,
    help="path to the directory where the tales will be written"
)


def main(args):

    prefix = args.sims_dir
    sim_set_prop_l = [
        SimSetProps(
            base_sim_name = '708cube_GasStaticG-1Einj',
            restart_sim_name = '708cube_GasStaticG-1Einj_restart-TIcool',
            restart_sim_kyr = 80.0 * 1000,
            prefix_path = prefix
        ),
        SimSetProps(
            base_sim_name='708cube_GasStaticG-1Einj',
            restart_sim_name='708cube_GasStaticG-1Einj_restartDelay-TIcool',
            restart_sim_kyr=130.0 * 1000,
            prefix_path=prefix
        ),
        SimSetProps(
            base_sim_name='708cube_GasStaticG-2Einj',
            restart_sim_name='708cube_GasStaticG-2Einj_restart-TIcool',
            restart_sim_kyr=80.0 * 1000,
            prefix_path=prefix
        )
    ]

    sne_rate_registry = {}
    for sim_set_prop in sim_set_prop_l:
        tmp = find_SNe_Rates(sim_set_prop, target_events_per_window=100)
        sne_rate_registry |= tmp

    if args.show_SNe_history_plot:
        _test_plot(sne_rate_registry)

    if args.out_dir is not None:
        print("writing SNe tables")
        out_dir = args.out_dir
        for sim_name, df in sne_rate_registry.items():
            sne_rate_registry[sim_name].to_csv(
                os.path.join(out_dir, f'{sim_name}.csv'), mode='w', 
            )
    else:
        print("skip writing of SNe tables (--out-dir not provided)")

    return 0

if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
