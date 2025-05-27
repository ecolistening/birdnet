import argparse
import multiprocessing as mp
import pathlib
import pandas as pd
import logging

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from tqdm import tqdm

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
)

from birdnet_multiprocessing.utils import chunked, suppress_output

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

analyzer = None
def init_worker():
    global analyzer
    with suppress_output():
        analyzer = Analyzer()

def species_probs(
    file_path: str,
    **kwargs: Any,
) -> pd.DataFrame:
    analyzer = Analyzer()
    return _species_probs(analyzer, file_path, **kwargs)

@suppress_output()
def _species_probs(
    analyzer: Analyzer,
    data: pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    recording = Recording(
        analyzer,
        str(data.file_path),
        lat=data.latitude,
        lon=data.longitude,
        date=data.timestamp.date(),
        **kwargs,
    )
    recording.analyze()
    df = pd.DataFrame(recording.detections)
    df["file_path"] = data.file_path
    return df

def batch_process_files(df: pd.DataFrame) -> pd.DataFrame:
    return [_species_probs(analyzer, row) for i, row in df.iterrows()]

def process_file(row: pd.Series) -> pd.DataFrame:
    global analyzer
    return _species_probs(analyzer, row)

def species_probs_multiprocessing(
    df: pd.DataFrame,
    num_workers: int,
    batch_size: int = 0,
) -> Iterable[pd.Series]:
    total = len(df)
    batched = batch_size > 1

    if batched:
        inputs = list(chunked(df, batch_size))
        fn = batch_process_files
    else:
        inputs = df
        fn = process_file

    with mp.Pool(processes=num_workers, initializer=init_worker) as map_pool:
        with tqdm(total=total, desc="Analysing...") as pbar:
            for results in map_pool.imap_unordered(fn, inputs):
                if batched:
                    for result in results:
                        yield result
                        pbar.update(1)
                else:
                    yield results
                    pbar.update(1)
