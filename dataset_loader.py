"""Load Kaggle job-post data from CSV files or ZIP archives."""

import io
import os
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd


MAX_CSV_SIZE = 200_000_000


def _csv_member(archive):
    csv_members = [
        item
        for item in archive.infolist()
        if not item.is_dir() and PurePosixPath(item.filename).suffix.lower() == ".csv"
    ]
    if not csv_members:
        raise ValueError("The ZIP archive does not contain a CSV file.")

    preferred = [
        item
        for item in csv_members
        if PurePosixPath(item.filename).name.lower() == "fake_job_postings.csv"
    ]
    if preferred:
        return preferred[0]
    if len(csv_members) == 1:
        return csv_members[0]
    raise ValueError("The ZIP archive contains multiple CSV files and no fake_job_postings.csv.")


def _read_archive(archive):
    member = _csv_member(archive)
    if member.file_size > MAX_CSV_SIZE:
        raise ValueError("The CSV inside the ZIP is larger than the 200 MB limit.")
    with archive.open(member) as csv_file:
        return pd.read_csv(csv_file)


def load_dataset_file(path):
    """Read a CSV or a ZIP containing a Kaggle CSV without extracting it."""
    dataset_path = Path(path)
    if dataset_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(dataset_path) as archive:
            return _read_archive(archive)
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError("Choose a .csv file or a .zip archive containing a CSV.")
    if dataset_path.stat().st_size > MAX_CSV_SIZE:
        raise ValueError("The CSV is larger than the 200 MB limit.")
    return pd.read_csv(dataset_path)


def load_uploaded_dataset(filename, content):
    """Read uploaded CSV or ZIP bytes without extracting archive members."""
    if filename.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                return _read_archive(archive)
        except zipfile.BadZipFile as error:
            raise ValueError("The uploaded ZIP archive could not be opened.") from error
    if not filename.lower().endswith(".csv"):
        raise ValueError("Upload a .csv file or a .zip archive containing a CSV.")
    if len(content) > MAX_CSV_SIZE:
        raise ValueError("The uploaded CSV is larger than the 200 MB limit.")
    return pd.read_csv(io.BytesIO(content))


def load_default_dataset():
    """Find configured or conventional local data, returning (frame, source) or None."""
    configured_path = os.environ.get("POSTMARK_DATASET_PATH")
    if configured_path:
        path = Path(configured_path).expanduser()
        if path.exists():
            return load_dataset_file(path), path.name

    project_csv = Path(__file__).with_name("fake_job_postings.csv")
    if project_csv.exists():
        return load_dataset_file(project_csv), project_csv.name

    downloads_archive = Path.home() / "Downloads" / "archive (4).zip"
    if downloads_archive.exists():
        return load_dataset_file(downloads_archive), "fake_job_postings.csv (from archive (4).zip)"
    return None