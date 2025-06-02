import click
import os
import time
import argparse
import pathlib
import logging
import pandas as pd
import re
import soundfile

from typing import Any

from birdnet_multiprocessing.utils import load_metadata_file, save_metadata_file
from birdnet_multiprocessing.main import embeddings_multiprocessing
from cli.utils import valid_data

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BIRDNET_INPUT_COLUMNS = ["file_path", "latitude", "longitude", "timestamp"]

__ALL__ = ["embed"]

@click.command(
    help="Embed audio features using BirdNET"
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
    "--num-workers",
    type=int,
    default=os.cpu_count(),
    help='Number of CPU cores, reverts to synchronous processing if 0 is specified'
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help='Batch size (audio files per worker)'
)
def main(
    audio_dir: pathlib.Path,
    index_file_name: str,
    save_dir: pathlib.Path,
    num_workers: int,
    batch_size: int,
    **kwargs: Any,
) -> None:
    """
    This command extracts BirdNET model embeddings from audio files referenced in
    the file index file saved at the root of --audio-dir. Results are persisted as a
    parquet file in the specified --save-dir directory.

    Example:
        python main.py embed --audio-dir=/path/to/audio/dir \
                             --index-file-name=metadata.parquet \
                             --save-dir=/path/to/saved/results
    """
    start_time = time.time()

    df = load_metadata_file(str(audio_dir / index_file_name))

    validate_columns(df.columns)

    df["file_path"] = df["file_path"].map(lambda file_path: str(audio_dir / file_path))

    df = valid_data(audio_dir, df)

    pending = embeddings_multiprocessing(
        df[df.columns.intersection(BIRDNET_INPUT_COLUMNS)],
        num_workers=num_workers,
        batch_size=batch_size,
        **kwargs
    )

    results_df = pd.concat(pending, axis=0).merge(
        df[["file_id", "file_path"]],
        left_on="file_path",
        right_on="file_path",
        how="left",
    )

    model_version = results_df.loc[0, "model"]
    save_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_parquet(save_dir / f"{model_version}_embeddings.parquet")

    log.info(f'Time taken: {time.time() - start_time} seconds')


