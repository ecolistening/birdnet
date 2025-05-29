import click
import pathlib
import pandas as pd
import re
import uuid

pattern = re.compile(r".*\.(wav|flac|mp3)$", re.IGNORECASE)

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
    help='path relative to --audio-dir specifying the location of the saved file index'
)
def main(
    audio_dir: pathlib.Path,
    index_file_name: str,
) -> None:
    df = pd.DataFrame([
        dict(file_path=str(file_path), uuid=str(uuid.uuid4()), file_name=file_path.name)
        for file_path in pathlib.Path(audio_dir).rglob('*.wav')
        if pattern.match(str(file_path))
    ])
    df.to_parquet(audio_dir / index_file_name)

if __name__ == "__main__":
    main()
