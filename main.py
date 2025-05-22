import argparse
import contextlib
import multiprocessing as mp
import pathlib
import pandas as pd
import os
import sys

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from typing import (
    Any,
    Callable,
    Iterable,
)

analyzer = None

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def map_reduce_parallel(
    inputs: Iterable,
    mapper: Callable,
    reducer: Callable,
    initializer: Callable | None = None,
    num_workers: int = mp.cpu_count()
) -> Iterable:
    collector = {}
    with mp.Pool(processes=num_workers, initializer=initializer) as map_pool:
        for key, value in tqdm(map_pool.imap_unordered(mapper, inputs), total=len(inputs), desc="Mapping"):
            collector[key] = value
    outputs = []
    with mp.Pool(processes=num_workers) as reduce_pool:
        for result in tqdm(reduce_pool.imap_unordered(reducer, collector.items()), total=len(collector), desc="Reducing"):
            outputs += result
    return outputs

def init_worker():
    global analyzer
    analyzer = Analyzer()

def mapper(item):
    global analyzer
    file_path, latitude, longitude, date = item.values()
    recording = Recording(analyzer, str(file_path), lat=latitude, lon=longitude, date=date)
    with suppress_output():
        recording.analyze()
    return (file_path, recording.detections)

def reducer(item):
    file_path, detections = item
    collection = defaultdict(float)
    for detection in detections:
        species_name = detection["scientific_name"]
        collection[species_name] = max(collection[species_name], detection["confidence"])
    return (file_path, collection)

def main(root_dir: pathlib.Path, num_workers: int):
    metadata = pd.read_parquet(root_dir / "metadata.parquet")
    metadata["date"] = metadata["timestamp"].dt.date
    metadata["file_path"] = metadata["file_name"].map(lambda file_name: root_dir / "data" / file_name)
    inputs = metadata[["file_path", "latitude", "longitude", "date"]].to_dict(orient="records")
    results = map_reduce_parallel(inputs, mapper, reducer, initializer=init_worker, num_workers=num_workers)
    import code; code.interact(local=locals())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively find all .wav files in a directory.")
    parser.add_argument(
        "--root-dir",
        type=lambda p: pathlib.Path(p), required=True,
        help="/path/to/data/directory"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes (default: CPU count)"
    )
    args = parser.parse_args()
    main(**vars(args))
