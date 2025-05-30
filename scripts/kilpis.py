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

AUDIO_FILE_REGEX = re.compile(r".*\.(wav|flac|mp3)$", re.IGNORECASE)

class Kilpis(Dataset):
    _FILE_NAME_REGEX = re.compile(r"^(SMA\d{5})_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:\b|[^0-9].*)?$")

    def to_datetime(self, file_name) -> Callable:
        return dt.datetime(*list(map(int, self._FILE_NAME_TO_DATETIME_REGEX.match(file_name).groups()))[:6])

    def to_site_name(self, file_path) -> Callable:
        import code; code.interact(local=locals())
        return "cairngorms/" + str(pathlib.Path(file_path).parent.parent.name).split("_")[1]

@click.command()
@click.option(
    "--audio-dir",
    required=True,
    type=lambda p: pathlib.Path(p),
    help="/path/to/audio/parent/directory"
)
def main(audio_dir):
    dataset = Kilpis(audio_dir)

if __name__ == "__main__":
    main()
