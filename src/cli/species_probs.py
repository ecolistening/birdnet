import click
import os
import time
import pathlib
import logging
import pandas as pd

from dask import config as cfg
from dask import dataframe as dd
from dask.distributed import Client
from typing import Any

from birdnet_dask import species_probs
from cli.utils import (
    load_metadata_file,
    save_metadata_file,
    validate_columns,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

cfg.set({
    "distributed.scheduler.worker-ttl": None
})

@click.command(
    help="Extract species probabilities using BirdNET",
)
@click.option(
    "--audio-dir",
    required=True,
    type=lambda p: pathlib.Path(p),
    help='/path/to/audio/parent/directory'
)
@click.option(
    "--index-file-name",
    default="metadata.parquet",
    type=str,
    help='path relative to --audio-dir specifying the location of the file index'
)
@click.option(
    "--save-dir",
    required=True,
    type=lambda p: pathlib.Path(p),
    help='/path/to/save/directory'
)
@click.option(
    "--min-conf",
    required=False,
    type=float,
    default=0.5,
    help='BirdNET confidence threshold'
)
@click.option(
    "--cores",
    type=int,
    default=os.cpu_count(),
    help='Number of CPU cores, reverts to synchronous processing if 0 is specified'
)
@click.option(
    "--memory",
    type=int,
    default=8,
    help='Upper bound of memory requirements in GiB'
)
@click.option(
    "--num-partitions",
    type=int,
    default=20,
    help='Number of data partitions'
)
@click.option(
    "--local-threads",
    type=int,
    default=1,
    help="",
)
@click.option(
    "--debug",
    type=bool,
    is_flag=True,
    default=False,
    help="Set synchronous for debugging",
)
def main(
    audio_dir: pathlib.Path,
    index_file_name: str,
    save_dir: pathlib.Path,
    min_conf: float,
    cores: int,
    memory: int,
    num_partitions: int,
    local_threads: int,
    debug: bool,
) -> None:
    """
    This command extracts all species probabilities from audio files referenced in
    the file index file saved at the root of --audio-dir. Results are persisted as a
    parquet file in the specified --save-dir directory.

    Example:
        python main.py species-probs --audio-dir=/path/to/audio/dir \
                                     --index-file-name=metadata.parquet \
                                     --save-dir=/path/to/saved/results \

    Note: Only persists species detections, does not contain references to
          files where no species were detected
    """

    df = load_metadata_file(str(audio_dir / index_file_name))
    validate_columns(df.columns)
    df["file_path"] = df["file_path"].map(lambda file_path: str(audio_dir / file_path))

    save_dir.mkdir(exist_ok=True, parents=True)

    start_time = time.time()

    if debug:
        cfg.set(scheduler='synchronous')

    client = Client(
        n_workers=cores,
        threads_per_worker=local_threads,
        memory_limit=f'{memory}GiB',
    )
    log.info(client)

    ddf = dd.from_pandas(df, npartitions=num_partitions)
    results_ddf = ddf.map_partitions(
        species_probs,
        min_conf=min_conf,
        meta=pd.DataFrame({
            "common_name": pd.Series(dtype="object"),
            "scientific_name": pd.Series(dtype="object"),
            "start_time": pd.Series(dtype="float64"),
            "end_time": pd.Series(dtype="float64"),
            "confidence": pd.Series(dtype="float64"),
            "label": pd.Series(dtype="object"),
            "path": pd.Series(dtype="object"),
            "file": pd.Series(dtype="object"),
            "min_conf": pd.Series(dtype="float64"),
            "model": pd.Series(dtype="object"),
        })
    )

    results_ddf.to_parquet(save_dir / f"birdnet_species_probs.parquet", write_index=True)

    client.close()

    log.info(f'Time taken: {time.time() - start_time} seconds')
