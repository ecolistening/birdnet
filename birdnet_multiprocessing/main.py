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

def process_audio_file_with_birdnet(
    file_path: str,
    latitude: float,
    longitude: float,
    date: dt.datetime,
) -> Tuple[str, Output]:
    global analyzer
    recording = Recording(analyzer, str(file_path), lat=latitude, lon=longitude, date=date)
    with suppress_output():
        recording.analyze()
    collection = defaultdict(float)
    for detection in recording.detections:
        species_name = detection["scientific_name"]
        collection[species_name] = max(collection[species_name], detection["confidence"])
    return file_path, collection

def process_files(items: List[Input]):
    return [process_audio_file_with_birdnet(**item) for item in items]

def process_audio_files_with_birdnet_mp(
    inputs: List[Input],
    num_workers: int
) -> List[Output]:
    with mp.Pool(processes=num_workers, initializer=init_worker) as map_pool:
        with tqdm(total=len(inputs), desc="Analysing...") as pbar:
            for results in map_pool.imap_unordered(process_files, inputs):
                for result in results:
                    yield result
                pbar.update(1)
