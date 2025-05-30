import click
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from cli.build_file_index import main as build_file_index
from cli.species_probs import main as species_probs
from cli.embed import main as embed
from cli.embeddings_and_species_probs import main as embeddings_and_species_probs

@click.group(
    help="""
Simple multiprocessing of audio using BirdNET

Please read the README for usage instructions
    """
)
def cli():
    pass

cli.add_command(build_file_index, name="build_file_index")
cli.add_command(species_probs, name="species-probs")
cli.add_command(embed, name="embed")
cli.add_command(embeddings_and_species_probs, name="embeddings-and-species-probs")

if __name__ == '__main__':
    cli()
