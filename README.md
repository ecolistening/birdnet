# Birdnet Multiprocessing

## Setup

Setup / install dependencies

```sh
uv venv
uv sync
source .venv/bin/activate
```

## Tests
Run the tests

```sh
pytest --audio-dir=/path/to/audio/dir
```

## Scripts

Extract species probabilities:
```sh
PYTHONPATH=src python -m scripts.sounding_out_chorus species-probs --audio-dir=/home/m4gpie/data/sounding_out_chorus --batch-size=6 --save-dir /home/m4gpie/data/
```

> **NB**: When no species are detected in a file, the resulting data frames will be **missing** references to these files.
>
> If making comparisons with ground truth labels, for example validating presence detection, you will need to (1) drop species specific duplicates (i.e. frame occurences), (2) pivot the table so each species has a unique column and (3) add rows for missing files with zeros as species probabilities.

Extract feature embeddings:
```sh
PYTHONPATH=src python -m scripts.sounding_out_chorus embed --audio-dir=/home/m4gpie/data/sounding_out_chorus --batch-size=6 --save-dir /home/m4gpie/data/
```
