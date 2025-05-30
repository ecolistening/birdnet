import click
import datetime as dt
import pathlib
import pandas as pd
import re
from typing import Callable

class Cairngorms:
    _FILE_NAME_TO_DATETIME_REGEX = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:\b|[^0-9].*)?$")

    def __init__(
        self,
        audio_dir: str | pathlib.Path,
        metadata_path: str | pathlib.Path,
        locations_path: str | pathlib.Path,
    ) -> None:
        self.audio_dir = audio_dir
        self.metadata = pd.read_parquet(metadata_path)
        self.locations = pd.read_parquet(locations_path)
        self.metadata["timestamp"] = self.metadata["file_name"].map(self.to_datetime)
        self.metadata["site"] = self.metadata["file_path"].map(self.to_site_name)

    @property
    def file_list_with_coords(self) -> pd.DataFrame:
        return self.metadata.merge(
            self.locations[["site", "latitude", "longitude"]],
            left_on="site",
            right_on="site",
        )

    def to_datetime(self, file_name) -> Callable:
        return dt.datetime(*list(map(int, self._FILE_NAME_TO_DATETIME_REGEX.match(file_name).groups()))[:6])

    def to_site_name(self, file_path) -> Callable:
        return "cairngorms/" + str(pathlib.Path(file_path).parent.parent.name).split("_")[1]


@click.command()
@click.option(
    "--audio-dir",
    required=True,
    type=lambda p: pathlib.Path(p),
    help="/path/to/audio/parent/directory"
)
@click.option(
    "--index-file-name",
    default="metadata.parquet",
    help="path relative to --audio-dir specifying the location of the saved file index. default: metadata.parquet"
)
@click.option(
    "--locations-path",
    required=True,
    help="path to locations.parquet file containing 'latitude', 'longitude' and a 'site' foreign key"
)
def main(audio_dir, index_file_name, locations_path):
    dataset = Cairngorms(audio_dir, audio_dir / index_file_name, locations_path)
    dataset.file_list_with_coords.to_parquet(audio_dir / index_file_name)

if __name__ == "__main__":
    main()
