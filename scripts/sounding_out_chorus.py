import os
import time
import argparse
import pathlib
import logging
import pandas as pd

from dask import config as cfg
from dask.distributed import Client, LocalCluster, progress

from birdnet_multiprocessing.client import setup_client
from birdnet_multiprocessing.main import _species_probs_multiprocessing

cfg.set({'distributed.scheduler.worker-ttl': None})

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main(
    audio_dir: str | pathlib.Path,
    save_dir: str | pathlib.Path | None,
    num_partitions: int,
    memory: int = 0,
    cores: int = 0,
    debug: bool = False,
) -> None:
    start_time = time.time()
    client = setup_client(memory, cores)
    log.info(client)
    log.info(f"{client.dashboard_link}")
    df = pd.read_parquet(audio_dir / "metadata.parquet")
    df["file_path"] = df["file_name"].map(lambda file_name: audio_dir / "data" / file_name)
    df = _species_probs_multiprocessing(df[["file_path", "latitude", "longitude", "timestamp"]], num_partitions=num_partitions)
    progress(df)
    if save_dir is not None:
        df.to_parquet(save_dir / "birdnet_predictions.parquet")
    log.info(f'Time taken: {time.time() - start_time} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract BirdNet predictions from audio files in parallel using Dask",
    )
    parser.add_argument(
        "--audio-dir",
        type=lambda p: pathlib.Path(p),
        help='/path/to/audio/parent/directory',
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        required=False,
        help='/path/to/save/directory',
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=os.cpu_count(),
        help='Number of CPU cores',
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=8,
        help='Memory per working (in GiB)',
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        help='Number of dataframe partitions',
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help='Flag indicating whether to run synchronously for debugging purposes'
    )
    main(**vars(parser.parse_args()))

