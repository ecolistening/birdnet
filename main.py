import click
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from cli.index_files import main as index_files
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

cli.add_command(index_files, name="index-files")
cli.add_command(species_probs, name="species-probs")
cli.add_command(embed, name="embed")
cli.add_command(embeddings_and_species_probs, name="embeddings-and-species-probs")

if __name__ == '__main__':
    cli()
