import click
import os
import time
import argparse
import pathlib
import logging
import pandas as pd
from typing import Any

from birdnet_multiprocessing.utils import chunked
from birdnet_multiprocessing.main import (
    species_probs_multiprocessing,
    embeddings_multiprocessing,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@click.group()
def cli():
    pass

@click.command()
@click.option( "--audio-dir", type=lambda p: pathlib.Path(p), help='/path/to/audio/parent/directory')
@click.option( "--save-dir", type=lambda p: pathlib.Path(p), help='/path/to/save/directory')
@click.option( "--num-workers", type=int, default=os.cpu_count(), help='Number of CPU cores')
@click.option( "--batch-size", type=int, default=6, help='Batch size (audio files per worker')
@click.option( "--min-conf", type=int, default=0.1, help='BirdNet confidence threshold')
@click.option( "--sync", type=bool, default=False, is_flag=True, help='Flag to use synchronous processing for syncging')
def species_probs(
    audio_dir: pathlib.Path,
    save_dir: pathlib.Path,
    num_workers: int,
    batch_size: int,
    **kwargs: Any,
) -> None:
    start_time = time.time()
    df = pd.read_parquet(audio_dir / "metadata.parquet")
    df["file_path"] = df["file_name"].map(lambda file_name: audio_dir / "data" / file_name)
    columns = ["file_path", "latitude", "longitude", "timestamp"]
    pending = species_probs_multiprocessing(df[columns], num_workers=num_workers, batch_size=batch_size, sync=sync, **kwargs)
    results_df = pd.concat(pending, axis=0)
    results_df["species_name"] = results_df["common_name"] + results_df["scientific_name"]
    save_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_parquet(save_dir / "birdnet_species_predict_proba.parquet")
    log.info(f'Time taken: {time.time() - start_time} seconds')

@click.command()
@click.option( "--audio-dir", type=lambda p: pathlib.Path(p), help='/path/to/audio/parent/directory')
@click.option( "--save-dir", type=lambda p: pathlib.Path(p), help='/path/to/save/directory')
@click.option( "--num-workers", type=int, default=os.cpu_count(), help='Number of CPU cores')
@click.option( "--batch-size", type=int, default=6, help='Batch size (audio files per worker')
@click.option( "--sync", type=bool, default=False, is_flag=True, help='Flag to use synchronous processing')
def embed(
    audio_dir: pathlib.Path,
    save_dir: pathlib.Path,
    num_workers: int,
    batch_size: int,
    sync: bool,
    **kwargs: Any,
) -> None:
    start_time = time.time()
    df = pd.read_parquet(audio_dir / "metadata.parquet")
    df["file_path"] = df["file_name"].map(lambda file_name: audio_dir / "data" / file_name)
    columns = ["file_path", "latitude", "longitude", "timestamp"]
    pending = embeddings_multiprocessing(df[columns], num_workers=num_workers, batch_size=batch_size, sync=sync, **kwargs)
    results_df = pd.concat(pending, axis=0)
    save_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_parquet(save_dir / "birdnet_embeddings.parquet")
    log.info(f'Time taken: {time.time() - start_time} seconds')

cli.add_command(species_probs)
cli.add_command(embed)

if __name__ == '__main__':
    cli()
