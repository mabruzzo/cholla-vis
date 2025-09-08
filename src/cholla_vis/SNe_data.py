# we probably want to shuffle things around...

import os

import pandas as pd
from .conf import PathConf
from .registry import _get_processed_data_dir

type SNeData = pd.DataFrame

def load_SNe_dataset(sim_name, *, path_conf: PathConf) -> SNeData:
    # load the supernova rate data
    data_dir_prefix = _get_processed_data_dir(path_conf)
    return pd.read_csv(
        os.path.join(data_dir_prefix, "SNe-rate-data", f"{sim_name}.csv"),
        index_col = 't_kyr'
    )
