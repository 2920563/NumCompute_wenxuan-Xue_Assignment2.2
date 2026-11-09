"""Chunk-wise CSV input without pandas or other data libraries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import numpy as np


def _validate_options(chunk_size: int, delimiter: str) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("delimiter must be a single character")


def _convert_rows(rows: list[list[str]], dtype: type, missing_values: tuple[str, ...]) -> np.ndarray:
    cleaned = [["nan" if value.strip() in missing_values else value.strip() for value in row] for row in rows]
    try:
        array = np.asarray(cleaned, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV contains a value that cannot be converted to the requested dtype") from exc
    if array.ndim != 2:
        raise ValueError("CSV rows must form a two-dimensional table")
    return array


def _split_target(array: np.ndarray, target_column: int | None):
    if target_column is None:
        return array
    column = int(target_column)
    if not -array.shape[1] <= column < array.shape[1]:
        raise ValueError("target_column is out of range")
    column %= array.shape[1]
    return np.delete(array, column, axis=1), array[:, column]


def iter_csv(
    path: str | Path,
    *,
    chunk_size: int = 128,
    target_column: int | None = -1,
    delimiter: str = ",",
    skip_header: bool = True,
    dtype: type = float,
    missing_values: tuple[str, ...] = ("", "NA", "NaN", "nan"),
) -> Iterator[np.ndarray | tuple[np.ndarray, np.ndarray]]:
    """Yield bounded chunks from a numeric CSV file.

    If ``target_column`` is not ``None``, each yield is ``(X_chunk, y_chunk)``.
    Otherwise each yield is the complete numeric chunk.
    """

    _validate_options(chunk_size, delimiter)
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        if skip_header:
            next(reader, None)
        rows: list[list[str]] = []
        width: int | None = None
        for line_number, row in enumerate(reader, start=2 if skip_header else 1):
            if not row or all(not value.strip() for value in row):
                continue
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError(f"inconsistent column count at line {line_number}")
            rows.append(row)
            if len(rows) == chunk_size:
                yield _split_target(_convert_rows(rows, dtype, missing_values), target_column)
                rows = []
        if rows:
            yield _split_target(_convert_rows(rows, dtype, missing_values), target_column)


def load_csv(path: str | Path, **kwargs):
    """Load all chunks from :func:`iter_csv` and concatenate them."""

    chunks = list(iter_csv(path, **kwargs))
    if not chunks:
        raise ValueError("CSV file contains no data rows")
    if isinstance(chunks[0], tuple):
        X = np.concatenate([chunk[0] for chunk in chunks], axis=0)
        y = np.concatenate([chunk[1] for chunk in chunks], axis=0)
        return X, y
    return np.concatenate(chunks, axis=0)


__all__ = ["iter_csv", "load_csv"]
