import click
import os
import time
import pathlib
import logging
import pandas as pd

from dask import config as cfg
from dask import bag as db
from dask.distributed import Client
from typing import Any

from birdnet_dask import (
    list_audio_files,
    valid_audio_file,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@click.command(
    help="Assign UUIDs to a file index, or build from scratch"
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
    help='path relative to --audio-dir specifying the location of the saved file index. default: metadata.parquet'
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
    cores: int,
    memory: int,
    local_threads: int,
    debug: bool,
) -> None:
    """
    This command builds a file index referencing all files within sub-folders of --audio-dir
    and checks they are valid audio files by attempting to stat using soundfile

    Example:
        python main.py index-files --audio-dir=/path/to/audio/dir --index-file-name=metadata.parquet
    """
    start_time = time.time()

    if debug:
        cfg.set(scheduler="synchronous")

    client = Client(
        n_workers=cores,
        threads_per_worker=local_threads,
        memory_limit=f"{memory}GiB",
    )
    log.info(client)

    b = db.from_sequence(list_audio_files(audio_dir))
    b = b.filter(valid_audio_file)

    ddf = b.to_dataframe(meta=pd.DataFrame({
        "file": pd.Series(dtype="object"),
        "path": pd.Series(dtype="object"),
    }))

    df = ddf.compute()
    df.to_parquet(audio_dir / index_file_name)
