import argparse
import datetime as dt
import multiprocessing as mp
import pathlib
import pandas as pd

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from collections import defaultdict
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

Input = Dict[str, float | dt.datetime]
Output = Dict[str | pathlib.Path, Dict[str, float]]

__ALL__ = [
    "process_audio_file_with_birdnet",
    "process_audio_files_with_birdnet_mp",
]

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

def _species_probs(
    analyzer: Analyzer,
    file_path: str,
    latitude: float | None = None,
    longitude: float | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    recording = Recording(analyzer, str(file_path), lat=latitude, lon=longitude, **kwargs)
    with suppress_output():
        recording.analyze()
    collection = defaultdict(float)
    df = pd.DataFrame(recording.detections)
    df["file_path"] = file_path
    if latitude is not None:
        df["latitude"] = latitude
    if longitude is not None:
        df["longitude"] = longitude
    return df

def batch_process_files(items: List[Input]) -> pd.DataFrame:
    return [_species_probs(analyzer, **item) for item in items]

def process_file(item: Input) -> pd.DataFrame:
    global analyzer
    return _species_probs(analyzer, **item)

def species_probs_multiprocessing(
    inputs: List[Input],
    num_workers: int
) -> List[Output]:
    batch_process = isinstance(inputs[0], list)
    fn = batch_process_files if batch_process else process_file
    with mp.Pool(processes=num_workers, initializer=init_worker) as map_pool:
        with tqdm(total=len(inputs), desc="Analysing...") as pbar:
            for results in map_pool.imap_unordered(fn, inputs):
                if batch_process:
                    for result in results:
                        yield result
                else:
                    yield results
                pbar.update(1)
