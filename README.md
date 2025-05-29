# BirdNet Multiprocessing
A wrapper around `birdnetlib` which has no opinion about directory structure for handling multiprocessing.

## Setup
Setup / install dependencies

```sh
uv venv
uv sync
source .venv/bin/activate
```

---

## Tests
Run the tests

```sh
pytest --audio-dir=/path/to/audio/dir
```

---

## Scripts
Extract species probabilities for each file:
```sh
PYTHONPATH=src python -m scripts.sounding_out_chorus species-probs --audio-dir=/home/m4gpie/data/sounding_out_chorus --batch-size=6 --save-dir /home/m4gpie/data/

```
> **NB**: Files with no detected species are missing from the results

Extract feature embeddings:
```sh
PYTHONPATH=src python -m scripts.sounding_out_chorus embed --audio-dir=/home/m4gpie/data/sounding_out_chorus --batch-size=6 --save-dir /home/m4gpie/data/
```
> **NB**: each row in the resulting table(s) correspond to 1024 features corresponding to a 3s frame

Extract both embeddings and species:
```sh
PYTHONPATH=src python -m scripts.sounding_out_chorus all --audio-dir=/home/m4gpie/data/sounding_out_chorus --batch-size=6 --save-dir /home/m4gpie/data/
```
> **NB**: Embeddings are padded with zero species probabilities for files with non-detections, so file sizes may be larger due to this passing, however this also accounts for species absence

---

## Using Singularity/ / Apptainer
Build the container:
```sh
singularity build --fakeroot app.sif app.def
```

Run the relevant script within the container:
```sh
singularity run -B /path/to/your/data:/data app.sif python -m scripts.sounding_out_chorus species-probs --audio-dir=/data --batch-size=6 --save-dir=/data
```
> **NB** If you want a custom save directory, you will need to specify that as a mount point
