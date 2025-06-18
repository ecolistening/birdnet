import click
import os
import time
import pathlib
import logging
import pandas as pd
import re
import soundfile
import uuid

from tqdm import tqdm

from cli.utils import (
    load_metadata_file,
    save_metadata_file,
    # valid_data
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

AUDIO_FILE_REGEX = re.compile(r".*\.(wav|flac|mp3)$", re.IGNORECASE)

def fetch_file_index(audio_dir):
    records = []
    pbar = tqdm(pathlib.Path(audio_dir).rglob('*'))
    for file_path in pbar:
        pbar.set_description("Recursively discovering audio files...")
        if AUDIO_FILE_REGEX.match(str(file_path)):
            records.append(dict(
                file_path=str(file_path),
                file_id=str(uuid.uuid4()),
                file_name=file_path.name
            ))
    return pd.DataFrame(records)

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
    "--num-workers",
    type=int,
    default=0,
    help='Number of CPU cores used during file validation, reverts to synchronous processing if 0 is specified'
)
def main(
    audio_dir: pathlib.Path,
    index_file_name: str,
    num_workers: int,
) -> None:
    """
    This command either:

    (1) Builds a file index referencing all files within sub-folders of --audio-dir with a UUID.

    (2) In the case where a specified file index already exists, it checks file_path is available
        when it is not, it defaults to (1), otherwise it appends a UUID

    Example:
        python main.py index-files --audio-dir=/path/to/audio/dir --index-file-name=metadata.parquet
    """
    if (audio_dir / index_file_name).exists():
        df = load_metadata_file(str(audio_dir / index_file_name))
        if "file_path" not in df.columns:
            df = fetch_file_index(audio_dir)
        else:
            if "file_id" not in df.columns:
                df["file_id"] = [uuid.uuid4() for i in range(len(df))]
    else:
        df = fetch_file_index(audio_dir)
        assert len(df) > 0, f"No files found in {str(audio_dir)}"

    df = valid_data(audio_dir, df, num_workers=num_workers)

    save_metadata_file(df, audio_dir / index_file_name)
