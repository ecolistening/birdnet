import os
import time
import argparse
import pathlib
import logging
import pandas as pd

from birdnet_multiprocessing.utils import chunked
from birdnet_multiprocessing.main import species_probs_multiprocessing

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main(
    audio_dir: str | pathlib.Path,
    save_dir: str | pathlib.Path | None,
    num_workers: int = 0,
    batch_size: int = 12,
) -> None:
    start_time = time.time()
    df = pd.read_parquet(audio_dir / "metadata.parquet")
    df["file_path"] = df["file_name"].map(lambda file_name: audio_dir / "data" / file_name)
    columns = ["file_path", "latitude", "longitude", "timestamp"]
    results_df = pd.concat(list(species_probs_multiprocessing(df[columns].iloc[:64], num_workers=num_workers, batch_size=batch_size)), axis=0)
    results_df["species_name"] = results_df["common_name"] + results_df["scientific_name"]
    # attach missing file names and pivot
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
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help='Number of CPU cores',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help='Batch size (audio files per worker)',
    )
    main(**vars(parser.parse_args()))

