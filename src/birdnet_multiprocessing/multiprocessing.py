import logging
import pandas as pd
import multiprocessing as mp

from birdnetlib.analyzer import Analyzer
from tqdm import tqdm
from typing import Any, Callable, Iterable, Iterator, Tuple

from birdnet_multiprocessing.utils import suppress_output

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

analyzer = None
@suppress_output()
def _init_worker():
    """
    Instantiate a single analyzer on each worker, cached globally.
    """
    global analyzer
    analyzer = Analyzer()

def process_file(fn: Callable, item: Tuple[int, pd.Series], **kwargs: Any) -> pd.DataFrame:
    i, row = item
    return fn(analyzer, row, **kwargs)

def process_batch(fn: Callable, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return pd.concat([
        fn(analyzer, row, **kwargs) for i, row in df.iterrows()
    ], join="outer").fillna(0)

def handle_processing(fn: Callable, inputs: Iterable) -> Iterable:
    log.info("Synchronous processing")
    return map(fn, inputs)

def handle_multiprocessing(fn: Callable, inputs: Iterable, **kwargs: Any) -> Iterator:
    log.info("Concurrent processing")
    with mp.Pool(**kwargs) as map_pool:
        for result in map_pool.imap(fn, inputs):
            yield result

def run_processing(
    fn: Callable,
    inputs: Iterable,
    num_workers: int = 0,
) -> Iterator:
    sync = num_workers == 0
    if sync:
        _init_worker()
        for result in handle_processing(fn, inputs):
            yield result
    else:
        pool_kwargs = dict(processes=num_workers, initializer=_init_worker)
        for result in handle_multiprocessing(fn, inputs, **pool_kwargs):
            yield result
