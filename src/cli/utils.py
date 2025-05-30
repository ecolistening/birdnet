import logging
import pandas as pd
import soundfile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def validate_columns(columns):
    assert "file_path" in columns, "'file_path' column must be specified"
    assert "uuid" in columns, "'uuid' column specifying a unique ID for each file must be specified"

def validate_all_audio(file_paths):
    valid_file_paths = []
    invalid_file_paths = []
    with tqdm(total=len(file_paths)) as pbar:
        pbar.set_description("Validating audio before processing...")
        for file_path in file_paths:
            try:
                soundfile.info(file_path)
                valid_file_paths.append(file_path)
            except soundfile.LibsndfileError as e:
                log.warning(e)
                invalid_file_paths.append(file_path)
            pbar.update(1)
    return valid_file_paths, invalid_file_paths

def valid_data(audio_dir, df):
    if (audio_dir / "failed_files.parquet").exists():
        invalid = pd.read_parquet(audio_dir / "failed_files.parquet")
        return df[~df.uuid.isin(invalid.uuid)]

    valid, invalid = validate_all_audio(df["file_path"])

    if len(invalid_df := df[df.file_path.isin(invalid)]):
        invalid_df.to_parquet(audio_dir / "failed_files.parquet")
        log.warning(f"Failed file references saved in '{audio_dir / 'failed_files.parquet'}'")

    return df[df.file_path.isin(valid)]
