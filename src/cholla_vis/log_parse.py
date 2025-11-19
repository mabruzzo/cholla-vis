# this file contains a bunch of logic for parsing Cholla logs
#
# Broadly speaking there are 2 parts to this machinery:
# 1. there is machinery for declaring regex patterns to detect logging information
# 2. there is the machinery that ties this logic together
#
# It definitely needs some cleanup (over time I tried a lot of different things and was
# very messy)

from collections.abc import Callable, Iterator, Sequence
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, auto
import itertools
import re
import typing

import numpy as np

@dataclass(frozen=True)
class StepSummaryInfo:
    n_step: int # specifies the number of steps that will have been completed
                # at the end of this current cycle
    sim_time: float # specifies the time at the end of the step
    sim_timestep : float # specifies the instantaneous timestep computed at
                         # the end of the step (will be used in the next step)
    timestep_time: float
    timestep_time_unit: str
    total_time: float
    total_time_unit: str

    @property
    def cycle_index(self):
        return self.n_step - 1

    @classmethod
    def try_from_line(cls, line):
        m = _try_linematch(line, _STANDARD_SUMMARY_PATTERN)
        if m is not None:
            return cls(
                n_step = int(m.group('nstep')),
                sim_time = float(m.group('simtime')),
                sim_timestep = float(m.group('simtstep')),
                timestep_time = float(m.group('tsteptime')),
                timestep_time_unit = m.group('tsteptimeU'),
                total_time = float(m.group('tottime')),
                total_time_unit = m.group('tottimeU'),
            )

type SNeHistoryBuilderCallback = Callable[
    [StepSummaryInfo | None, StepSummaryInfo, int, int], None
]
type PreSummaryLogLines = list[str]

# Historically the following block of logic was handled somewhat separately
# from the rest of the file
# ======================================================================
# NOTE:
# - there is a problem that GPU warning printf statements are sometimes
#   printed in the middle of another statement (doesn't come up unless
#   we are printing a LOT)
#
#
# Unfortunately, I've duplicated the following logic in a LOT of places

""" # the following class isn't actually used, but would simplify things!
class PeekableItrWrapper: 
    def __init__(self, itr):
        self._wrapped_itr = iter(itr)
        self._current_peek_val = []

    def __iter__(self): return self

    def peek(self, *fallback_arg):
        # fallback argument is optional. It will NOT affect the next value 
        # returned by this method or __next__ if the iterator is exhausted
        if len(fallback_arg) > 1:
            raise RuntimeError("peek can only be called with up to 1 arg")

        if not self._current_peek_val:
            try:
                self._current_peek_val.append(next(self._wrapped_itr))
            except StopIteration:
                if fallback_arg: return fallback_arg[0]
                raise StopIteration
        return self._current_peek_val[0]

    def __next__(self):
        if self._current_peek_val: return self._current_peek_val.pop()
        return next(self._wrapped_itr)
"""

def _try_linematch(line, pattern):
    if line is None:
        return None
    return pattern.fullmatch(line)

_FLOAT_E_EXPR = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'

class _NamedPattern: # the class acts as a namespace
    @staticmethod
    def signed_int(name: str) -> str:
        return f'(?P<{name}>[-+]?\\d+)'

    @staticmethod
    def unsigned_int(name: str) -> str:
        return f'(?P<{name}>\\d+)'

    @staticmethod
    def float(name: str) -> str:
        return f'(?P<{name}>{_FLOAT_E_EXPR})'

    @staticmethod
    def float_unit(val_name: str, u_name: str) -> str:
        return f'(?P<{val_name}>{_FLOAT_E_EXPR})[ ]+(?P<{u_name}>[a-zA-Z/]+)'


_FEEDBACK_LINE_PATTERN = re.compile('feedback: ' + r",[ ]*".join([
    r"time[ ]+" + _NamedPattern.float("time"),
    r"dt=[ ]*" +  _NamedPattern.float("dt"),
    r"vrms =[ ]*" + _NamedPattern.float_unit('vrms', 'vrmsU'),
][:]))


@dataclass(frozen=True)
class FeedbackSummaryInfo:
    time: float
    dt: float
    vrms: float
    vrms_unit: str

    @classmethod
    def try_from_line(cls, line):
        m = _try_linematch(line, _FEEDBACK_LINE_PATTERN)
        if m is not None:
            return cls(time = float(m.group('time')), dt = float(m.group('dt')),
                       vrms = float(m.group('vrms')), vrms_unit = m.group('vrmsU'))


_STANDARD_SUMMARY_PATTERN = re.compile(r"[ ]+".join([
    r"n_step:[ ]*(?P<nstep>\d+)",
    r"sim time:[ ]*" + _NamedPattern.float("simtime"),
    r"sim timestep:[ ]*" + _NamedPattern.float("simtstep"),
    r"timestep time =[ ]*" + _NamedPattern.float_unit('tsteptime', 'tsteptimeU'),
    r"total time =[ ]*" + _NamedPattern.float_unit('tottime', 'tottimeU'),
][:])+ r"[ \t]*")

# ======================================================================
# Historically the preceeding logic was separated from the following logic:


# this type is used to hold all of the initially parsed cycle properties
# -> we actually build this up gradually
# -> if we ever shifted to a "plugin" system for parsing, it probably won't
#    be possible to declare every key
CyclePropsParseDict = typing.TypedDict(
    "ParsedCycleProps",
    {
        "cummulative resolved SN": int,
        "cummulative unresolved SN": int,
        "summary": StepSummaryInfo,
        "feedback summary": FeedbackSummaryInfo,
        "cur_sn_list2": list[typing.Any],

        # the following entry may or may not exist
        "avg-slow-cell-line": str,
    },
    total = False
)


_CUM_SN_COUNT_PATTERN = re.compile('    cummulative: ' + r',[ ]*'.join([
    r'#SN:[ ]*' + _NamedPattern.unsigned_int('total'),
    r'ratio of resolved \(R:[ ]*' +  _NamedPattern.unsigned_int('resolved'),
    r'UR:[ ]*' + _NamedPattern.unsigned_int('unresolved'),
]) + r'\)[ ]*=[ ]*' + _NamedPattern.float("ratio"))

def try_cummulative_SN_count(line: str, cur_props: CyclePropsParseDict) -> bool:
    m = _try_linematch(line, _CUM_SN_COUNT_PATTERN)
    if m is not None:
        cur_props['cummulative resolved SN'] = int(m.group('resolved'))
        cur_props['cummulative unresolved SN'] = int(m.group('unresolved'))
        return True
    return False

def try_avgcell_line(line: str, cur_props: CyclePropsParseDict):
    if (line is not None) and line.startswith(' Average Slow Cell '):
        cur_props['avg-slow-cell-line'] = line
        return True
    return False

class Status(Enum):
    FOUND_SUMMARY = auto()
    MATCH = auto()
    NOMATCH = auto()

def _parse_helper(line: str, cur_props: CyclePropsParseDict) -> Status:
    # used to extract some useful info
    if try_cummulative_SN_count(line, cur_props):
        return Status.MATCH
    elif try_avgcell_line(line, cur_props):
        return Status.MATCH

    summary_info = StepSummaryInfo.try_from_line(line)
    if summary_info is None:
        return Status.NOMATCH
    else:
        cur_props['summary'] = summary_info
        return Status.FOUND_SUMMARY


_SAVING_SNAP_PATTERN = re.compile(
    r'Saving Snapshot:[ ]+' + _NamedPattern.unsigned_int('num')
)

# the leading space is intentional
_SAVING_PARTICLE_PATTERN = re.compile(
    r' Total Particles:[ ]+' + _NamedPattern.unsigned_int('count')
)


def _consume_until_nonspace(line_iter: Iterator[str]) -> str:
    while True:
        l = next(line_iter, None)
        if (l is None) or ((len(l) > 0) and not l.isspace()):
            return l

def _get_sn_detail_chunks(lines: Sequence[str]) -> list[str]:
    # this is crude! (and I would love to parse all of the contained information!)
    chunk_l = []

    i = 0
    while i < len(lines):
        if lines[i].startswith('...(block='):
            cur_chunk = [lines[i]]
            while (i+1)<len(lines) and lines[i+1].startswith('    '):
                cur_chunk.append(lines[i+1])
                i+=1
            chunk_l.append(cur_chunk)
        i+=1
    return chunk_l

def _get_sn_announcement(lines: list[str]) -> list[dict[str, typing.Any]]:
    out = []
    for line in lines:
        if line.startswith('..fb: {'):
            out.append(json.loads(line[5:]))
    return out

def cycle_chunk_itr(
    line_iter: Iterator[str],
    last_read_cycle: int = 0,
    has_feedback_summary: bool = True
) -> Iterator[tuple[CyclePropsParseDict, PreSummaryLogLines]]:
    # yields pairs for each chunk of lines corresponding to a single cycle
    # pair[0] is a dictionary containing some summary info
    # pair[1] is a list of lines before the primary summary info

    lookahead_line = next(line_iter, None)
    while lookahead_line is not None:
        pre_summary_lines, step_summary = [], None
        cur_props = {}

        if not lookahead_line.isspace():
            if _parse_helper(lookahead_line, cur_props) == Status.NOMATCH:
                pre_summary_lines.append(lookahead_line)

        if 'summary' not in cur_props:
            for line in line_iter:
                status = _parse_helper(line, cur_props)
                if status == Status.NOMATCH:
                    pre_summary_lines.append(line)
                elif status == Status.FOUND_SUMMARY:
                    break

        if 'summary' in cur_props: # always execute unless the log was truncated
            if (last_read_cycle + 1) != cur_props['summary'].n_step:
                raise AssertionError(
                    f"last_read_cycle+1 ({last_read_cycle+1}) must equal "
                    f"cur_props['summary'].n_step, ({cur_props['summary'].n_step})"
                )

        # after the step-summary we possibly check for Average-Slow-Cell,
        # Feedback-Summary and Output Info. Anything else is part of the next chunk
        lookahead_line = _consume_until_nonspace(line_iter)

        if try_avgcell_line(lookahead_line, cur_props):
            lookahead_line = _consume_until_nonspace(line_iter)

        if has_feedback_summary and (lookahead_line is not None):
            tmp = FeedbackSummaryInfo.try_from_line(lookahead_line)
            if tmp is None:
                raise RuntimeError(
                    "expected the next non-empty line after the summary to be"
                    f"feedback-summary. Instead, found: {lookahead_line!r}")
            cur_props['feedback summary'] = tmp
            lookahead_line = _consume_until_nonspace(line_iter)

        m = _try_linematch(lookahead_line, _SAVING_SNAP_PATTERN)
        if m is not None:
            cur_props['saved_snap_num'] = int(m.group('num'))
            lookahead_line = next(line_iter,None)
            # check for record of number of saved particles the next line when we know
            # we recoreded an output
            m2 = _try_linematch(lookahead_line, _SAVING_PARTICLE_PATTERN)
            if m2 is not None:
                cur_props['saved_particle_count'] = int(m2.group('count'))
                lookahead_line = next(line_iter, None)

        if True:
            cur_props['cur_sn_list2'] = []
        else:
            cur_props['cur_sn_list2'] = _get_sn_announcement(pre_summary_lines)

        yield cur_props, pre_summary_lines

        last_read_cycle = cur_props['summary'].n_step

class PrecycleData(typing.TypedDict):
    start_cycle: int
    start_time: float
    macro: str
    commit: str

class Parser:

    def __init__(self):
        self._precycle_data = {}

    def _exec_precycle_parser(self,line_iter):
        starting_cycle_time_pair = None

        pattern = re.compile(r'^Nstep = (\d+)[ ]+Simulation time = (' + _FLOAT_E_EXPR + ')[ ]*$')

        name_pattern_pairs = [('macro', re.compile(r"^Macro Flags     =(.+)")),
                              ('commit', re.compile(r"^Git Commit Hash = (.+)"))]

        for line in line_iter:
            m = pattern.match(line)
            if m is not None:
                # recall m[0] will match full line
                self._precycle_data['start_cycle'] = int(m[1])
                self._precycle_data['start_time'] = float(m[2])
                break

            for name, cur_pattern in name_pattern_pairs:
                m = cur_pattern.match(line)
                if m is not None:
                    assert name not in self._precycle_data
                    self._precycle_data[name] = m[1].strip()

        for line in line_iter:
            if line == 'Starting calculations.':
                break

    def precycle_data(self) -> PrecycleData:
        return self._precycle_data

    def __call__(
        self, f: str | typing.TextIO, has_feedback_summary: bool = True
    ) -> Iterator[tuple[CyclePropsParseDict, PreSummaryLogLines]]:
        """iterate over sections of the specified log file"""

        # in the future, we may want to allow for selecting the information we
        # are interested in

        if isinstance(f, str):
            cm = open(f, "r")
        else:
            cm = nullcontext(f)
        
        with cm as _f:
            line_iter = (line[:-1] for line in _f)
            self._exec_precycle_parser(line_iter)
            yield from cycle_chunk_itr(
                line_iter,
                last_read_cycle = self._precycle_data['start_cycle'],
                has_feedback_summary = has_feedback_summary
            )

def get_previouscompletedsteps_and_starttime(f):

    if isinstance(f, str):
        cm = open(f, "r")
    else:
        cm = nullcontext(f)

    parser = Parser()
    with cm as _f:
        _ = next(parser(_f))
    tmp = parser.precycle_data()
    return tmp['start_cycle'], tmp['start_time']

def gather_summary(
    path: str,
    stop_cycle_index: int = float('inf'),
    sne_history_builder: SNeHistoryBuilderCallback | None = None
) -> tuple[dict[str, np.ndarray], PrecycleData]:
    """
    Gathers summary information from a single log file
    """

    p = Parser()

    expected = [
        "t", "dt", "cum_resolved_sn_count", "cum_unresolved_sn_count",
        "avg_slow_cell_i", "sn_l2"
    ]

    tmp = {key: [] for key in expected}

    print(f"Begin Parsing: {path}")
    try:
        last_summary = None
        last_cum_resolved_sn_count = 0
        last_cum_unresolved_sn_count = 0
        for i,(props,lines) in enumerate(p(path)):
            cur_cycle_index = props['summary'].cycle_index
            assert (
                (i > 0) or
                (stop_cycle_index is None) or
                cur_cycle_index < stop_cycle_index
            )

            tmp["t"].append(props['summary'].sim_time)
            tmp["dt"].append(props['summary'].sim_timestep)
            tmp["cum_resolved_sn_count"].append(props['cummulative resolved SN'])
            tmp["cum_unresolved_sn_count"].append(props['cummulative unresolved SN'])
            if 'avg-slow-cell-line' in props:
                tmp["avg_slow_cell_i"].append(i)

            cur_resolved_sn_count = (
                tmp["cum_resolved_sn_count"][-1] - last_cum_resolved_sn_count
            )
            cur_unresolved_sn_count = (
                tmp["cum_unresolved_sn_count"][-1] - last_cum_unresolved_sn_count
            )
            assert cur_resolved_sn_count >= 0
            assert cur_unresolved_sn_count >= 0

            if len(props['cur_sn_list2']) > 0:
                l = [e for e in props['cur_sn_list2']]
                tmp["sn_l2"] += l

            if (
                (sne_history_builder is not None) and
                ((cur_resolved_sn_count > 0) or
                 (cur_unresolved_sn_count > 0))
            ):
                sne_history_builder(
                    prev_step_props=last_summary,
                    cur_step_props=props['summary'],
                    resolved_sn_count = cur_resolved_sn_count,
                    cur_unresolved_sn_count = cur_unresolved_sn_count
                )
                

            last_summary = props['summary']
            last_cum_resolved_sn_count = tmp["cum_resolved_sn_count"][-1]
            last_cum_unresolved_sn_count = tmp["cum_unresolved_sn_count"][-1]
            if (cur_cycle_index + 1) == stop_cycle_index:
                break
    except KeyError:
        # expect a keyerror at the end
        pass

    data = {k: np.array(v) for k,v in tmp.items()}
    return data, p.precycle_data()


if __name__ == "__main__":

    print(
        FeedbackSummaryInfo.try_from_line(
            'feedback: time 10.000000, dt=10.000000, vrms = 24.155260 km/s'
        )
    )

    print(
        StepSummaryInfo.try_from_line(
            'n_step: 1   sim time:  3.0199755   sim timestep: 3.0200e+00  timestep time =   749.973 ms   total time =   29.3605 s'
        )
    )

    try_cummulative_SN_count(
        '    cummulative: #SN: 1, ratio of resolved (R: 1, UR: 0) = 1.000e+00',
        {}
    )


