import click
import os
import time
import argparse
import pathlib
import logging
import pandas as pd
import re
import sys
import uuid

from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from birdnet_multiprocessing.utils import load_metadata_file, save_metadata_file
from birdnet_multiprocessing.main import (
    species_probs_multiprocessing,
    embeddings_multiprocessing,
    embeddings_and_species_probs_multiprocessing,
)

BIRDNET_INPUT_COLUMNS = ["file_path", "latitude", "longitude", "timestamp"]

AUDIO_FILE_REGEX = re.compile(r".*\.(wav|flac|mp3)$", re.IGNORECASE)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def validate_columns(columns):
    assert "file_path" in columns, "'file_path' column must be specified"
    assert "uuid" in columns, "'uuid' column specifying a unique ID for each file must be specified"

# --------------------------------------------------------------------------------------------- #

@click.command()
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
def build_file_index(
    audio_dir: pathlib.Path,
    index_file_name: str,
) -> None:
    df = pd.DataFrame([
        dict(file_path=str(file_path), uuid=str(uuid.uuid4()), file_name=file_path.name)
        for file_path in pathlib.Path(audio_dir).rglob('*.wav')
        if AUDIO_FILE_REGEX.match(str(file_path))
    ])
    save_metadata_file(df, audio_dir / index_file_name)

# --------------------------------------------------------------------------------------------- #

@click.command()
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
@click.option(
    "--min-conf",
    required=False,
    type=int,
    default=0.5,
    help='BirdNET confidence threshold'
)
def species_probs(
    audio_dir: pathlib.Path,
    index_file_name: str,
    save_dir: pathlib.Path,
    num_workers: int,
    batch_size: int,
    **kwargs: Any,
) -> None:
    start_time = time.time()

    df = load_metadata_file(str(audio_dir / index_file_name))

    validate_columns(df.columns)

    df["file_path"] = df["file_path"].map(lambda file_path: str(audio_dir / file_path))

    pending = species_probs_multiprocessing(
        df[df.columns.intersection(BIRDNET_INPUT_COLUMNS)],
        num_workers=num_workers,
        batch_size=batch_size,
        **kwargs
    )

    results_df = pd.concat(pending, axis=0).merge(
        df[["file_path", "uuid"]],
        left_on="file_path",
        right_on="file_path",
        how="left",
    )

    model_version = results_df.loc[0, "model"]
    save_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_parquet(save_dir / f"{model_version}_species_probs.parquet")

    log.info(f'Time taken: {time.time() - start_time} seconds')

# --------------------------------------------------------------------------------------------- #

@click.command()
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
def embed(
    audio_dir: pathlib.Path,
    index_file_name: str,
    save_dir: pathlib.Path,
    num_workers: int,
    batch_size: int,
    **kwargs: Any,
) -> None:
    start_time = time.time()

    df = load_metadata_file(str(audio_dir / index_file_name))

    validate_columns(df.columns)

    df["file_path"] = df["file_path"].map(lambda file_path: str(audio_dir / file_path))

    pending = embeddings_multiprocessing(
        df[df.columns.intersection(BIRDNET_INPUT_COLUMNS)],
        num_workers=num_workers,
        batch_size=batch_size,
        **kwargs
    )

    results_df = pd.concat(pending, axis=0).merge(
        df[["file_path", "uuid"]],
        left_on="file_path",
        right_on="file_path",
        how="left",
    )

    model_version = results_df.loc[0, "model"]
    save_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_parquet(save_dir / f"{model_version}_embeddings.parquet")

    log.info(f'Time taken: {time.time() - start_time} seconds')

# --------------------------------------------------------------------------------------------- #

@click.command()
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
@click.option(
    "--min-conf",
    required=False,
    type=int,
    default=0.5,
    help='BirdNet confidence threshold'
)
def embeddings_and_species_probs(
    audio_dir: pathlib.Path,
    index_file_name: str,
    save_dir: pathlib.Path,
    num_workers: int,
    batch_size: int,
    **kwargs: Any,
) -> None:
    start_time = time.time()

    df = load_metadata_file(str(audio_dir / index_file_name))

    validate_columns(df.columns)

    df["file_path"] = df["file_path"].map(lambda file_path: str(audio_dir / file_path))

    pending = embeddings_and_species_probs_multiprocessing(
        df[df.columns.intersection(BIRDNET_INPUT_COLUMNS)],
        num_workers=num_workers,
        batch_size=batch_size,
        **kwargs
    )

    results_df = pd.concat(pending, join="outer", axis=0).fillna(0)
    results_df = results_df.merge(
        df[["file_path", "uuid"]],
        left_on="file_path",
        right_on="file_path",
        how="left",
    )

    model_version = results_df.loc[0, "model"]
    save_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_parquet(save_dir / f"{model_version}_embeddings_and_species_probs.parquet")

    log.info(f'Time taken: {time.time() - start_time} seconds')


# --------------------------------------------------------------------------------------------- #

@click.group()
def cli():
    pass

cli.add_command(build_file_index)
cli.add_command(species_probs)
cli.add_command(embed)
cli.add_command(embeddings_and_species_probs)

if __name__ == '__main__':
    cli()
