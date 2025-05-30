import click
import datetime as dt
import logging
import pathlib
import pandas as pd
import re
import sys

from typing import Callable

from scripts.dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Cairngorms(Dataset):
    _FILE_NAME_TO_DATETIME_REGEX = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:\b|[^0-9].*)?$")

    def to_datetime(self, file_name) -> Callable:
        return dt.datetime(*list(map(int, self._FILE_NAME_TO_DATETIME_REGEX.match(file_name).groups()))[:6])

    def to_site_name(self, file_path) -> Callable:
        return "cairngorms/" + str(pathlib.Path(file_path).parent.parent.name).split("_")[1]

    def _build_locations(self):
        # NB: currently provided to us, but could be ingested somehow...
        return pd.read_parquet(self.audio_dir / self._LOCATIONS_INDEX)

    def _parse_recorder_config_file(file_path) -> Dict[str, str]:
        config = {}
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                    config[key.strip()] = value.strip()
        return config


@click.command()
@click.option(
    "--audio-dir",
    required=True,
    type=lambda p: pathlib.Path(p),
    help="/path/to/audio/parent/directory"
)
def main(audio_dir):
    Cairngorms(audio_dir)
    print(dataset.metadata.to_markdown())

if __name__ == "__main__":
    main()
