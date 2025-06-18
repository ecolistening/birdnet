import click
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from cli.index_files import main as index_files
from cli.species_probs import main as species_probs
from cli.embed import main as embed

@click.group(
    help="""
Process audio using Dask and BirdNET

Please read the README for usage instructions
    """
)
def cli():
    pass

cli.add_command(index_files, name="index-files")
cli.add_command(species_probs, name="species-probs")
cli.add_command(embed, name="embed")

if __name__ == '__main__':
    cli()
