# BirdNet Multiprocessing

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

Extract species probabilities for each file (note that files with no detected species are missing from the results):
```sh
PYTHONPATH=src python -m scripts.sounding_out_chorus species-probs --audio-dir=/home/m4gpie/data/sounding_out_chorus --batch-size=6 --save-dir /home/m4gpie/data/
```

Extract feature embeddings (each row corresponds to 1024 features describing a 3s frame):
```sh
PYTHONPATH=src python -m scripts.sounding_out_chorus embed --audio-dir=/home/m4gpie/data/sounding_out_chorus --batch-size=6 --save-dir /home/m4gpie/data/
```

Extract both embeddings and species (note that embeddings are padded with zero species probabilities for files with non-detections):
```sh
PYTHONPATH=src python -m scripts.sounding_out_chorus all --audio-dir=/home/m4gpie/data/sounding_out_chorus --batch-size=6 --save-dir /home/m4gpie/data/
```
