import click
import datetime as dt
import logging
import pathlib
import pandas as pd
import re
import sys
import uuid

from typing import Callable, ClassVar

from scripts.dataset import Dataset, Site

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Kilpis(Dataset):
    _DATETIME_REGEX: ClassVar[re.Pattern] = re.compile(r"^(?:\b|[^0-9].*)?(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:\b|[^0-9].*)?$")
    _SITE_REGEX: ClassVar[re.Pattern] = re.compile(r"^([A-Z]{3}\d{5})(?:\b|[^0-9].*)?$")

    def to_datetime(self, file_name) -> Callable:
        datetime_fields = self._DATETIME_REGEX.match(file_name).groups()
        return dt.datetime(*list(map(int, datetime_fields))[:6])

    def to_site_name(self, file_path) -> Callable:
        return self._SITE_REGEX.match(pathlib.Path(file_path).name).group(1)

    def _load_locations(self):
        if (self.audio_dir / self._LOCATIONS_INDEX).exists():
            return pd.read_parquet(self.audio_dir / self._LOCATIONS_INDEX)

        def summary_file_to_coordinates(row):
            return dict(
                latitude=row["LAT"] if row["Unnamed: 3"] == "N" else -row["LAT"],
                longitude=row["LON"] if row["Unnamed: 5"] == "E" else -row["LON"],
            )

        recorder_id_regex = re.compile("^([A-Z]{3}\d{5})_Summary.txt$")
        recorders = []
        for file_path in pathlib.Path(self.audio_dir).rglob('*'):
            if (match := recorder_id_regex.match(file_path.name)):
                # NB: assume the recorders are in a fixed location
                # this can easily be amended to use the date time available
                # in the summary file to determine when the recorder changed
                # location (presuming its correct)
                recorders.append(Site(
                    site_name=match.group(1),
                    site_id=str(uuid.uuid4()),
                    **summary_file_to_coordinates(next(pd.read_csv(file_path, chunksize=1)).loc[0]),
                ))
        recorders = pd.DataFrame(recorders)
        recorders.to_parquet(self.audio_dir / self._LOCATIONS_INDEX)
        return recorders

@click.command()
@click.option(
    "--audio-dir",
    required=True,
    type=lambda p: pathlib.Path(p),
    help="/path/to/audio/parent/directory"
)
def main(audio_dir):
    dataset = Kilpis(audio_dir)
    print(dataset.metadata)

if __name__ == "__main__":
    main()
