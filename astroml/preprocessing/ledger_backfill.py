"""Polars-based preprocessing for large ledger backfill datasets.

This module is designed for backfills with millions of rows. It keeps work in
Polars lazy expressions end-to-end so data can be streamed from source files to
columnar output with low memory overhead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import polars as pl

BackfillFormat = Literal["parquet", "csv", "ndjson", "jsonl"]


def _col_or_null(name: str, existing: set[str]) -> pl.Expr:
    if name in existing:
        return pl.col(name)
    return pl.lit(None)


def _infer_input_format(path: Path) -> BackfillFormat:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "csv"
    if suffix in (".ndjson", ".jsonl"):
        return "ndjson"
    if path.is_dir():
        if list(path.glob("*.parquet")):
            return "parquet"
        if list(path.glob("*.csv")):
            return "csv"
        if list(path.glob("*.ndjson")) or list(path.glob("*.jsonl")):
            return "ndjson"
    raise ValueError(
        f"Unable to infer input format for '{path}'. Use input_format explicitly."
    )


def _resolve_scan_target(path: Path, patterns: Iterable[str]) -> str:
    if path.is_file():
        return str(path)

    for pattern in patterns:
        if list(path.glob(pattern)):
            return str(path / pattern)
    raise FileNotFoundError(
        f"No files matched {tuple(patterns)} in input directory '{path}'."
    )


def scan_backfill_dataset(
    input_path: str | Path,
    input_format: BackfillFormat | None = None,
) -> pl.LazyFrame:
    """Scan a backfill dataset using Polars lazy APIs.

    Args:
        input_path: File or directory containing backfill rows.
        input_format: Optional explicit format. If omitted, inferred from path.

    Returns:
        Polars ``LazyFrame`` that can be transformed further without immediate
        materialization.
    """
    path = Path(input_path)
    fmt = input_format or _infer_input_format(path)
    normalized_fmt = "ndjson" if fmt == "jsonl" else fmt

    if normalized_fmt == "parquet":
        return pl.scan_parquet(_resolve_scan_target(path, ("*.parquet",)))
    if normalized_fmt == "csv":
        return pl.scan_csv(_resolve_scan_target(path, ("*.csv",)))
    if normalized_fmt == "ndjson":
        return pl.scan_ndjson(_resolve_scan_target(path, ("*.ndjson", "*.jsonl")))
    raise ValueError(f"Unsupported input format: {fmt}")


def preprocess_ledger_backfill(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize raw backfill rows into a typed transaction-like dataset."""
    schema_names = set(frame.collect_schema().names())

    op_type = _col_or_null("type", schema_names).cast(pl.String)
    sender = pl.coalesce(
        [
            _col_or_null("source_account", schema_names),
            _col_or_null("from", schema_names),
            _col_or_null("funder", schema_names),
        ]
    ).cast(pl.String)
    receiver = pl.coalesce(
        [
            _col_or_null("to", schema_names),
            pl.when(op_type == "create_account")
            .then(_col_or_null("account", schema_names))
            .otherwise(None),
            pl.when(op_type == "account_merge")
            .then(_col_or_null("into", schema_names))
            .otherwise(None),
            _col_or_null("destination_account", schema_names),
        ]
    ).cast(pl.String)
    amount = (
        pl.coalesce(
            [
                _col_or_null("amount", schema_names),
                _col_or_null("starting_balance", schema_names),
                _col_or_null("destination_amount", schema_names),
                _col_or_null("source_amount", schema_names),
            ]
        )
        .cast(pl.Float64, strict=False)
        .alias("amount")
    )
    asset_code = _col_or_null("asset_code", schema_names).cast(pl.String)
    asset_issuer = _col_or_null("asset_issuer", schema_names).cast(pl.String)
    asset_type = _col_or_null("asset_type", schema_names).cast(pl.String)
    asset = (
        pl.when(asset_type == "native")
        .then(pl.lit("XLM"))
        .when(asset_code.is_not_null() & asset_issuer.is_not_null())
        .then(pl.concat_str([asset_code, pl.lit(":"), asset_issuer]))
        .when(asset_code.is_not_null())
        .then(asset_code)
        .otherwise(pl.lit("UNKNOWN"))
        .alias("asset")
    )
    ledger_sequence = (
        pl.coalesce(
            [
                _col_or_null("ledger_sequence", schema_names),
                _col_or_null("ledger", schema_names),
                _col_or_null("sequence", schema_names),
            ]
        )
        .cast(pl.Int64, strict=False)
        .alias("ledger_sequence")
    )
    raw_timestamp = pl.coalesce(
        [
            _col_or_null("created_at", schema_names),
            _col_or_null("closed_at", schema_names),
            _col_or_null("timestamp", schema_names),
        ]
    )
    timestamp = (
        pl.when(raw_timestamp.is_null())
        .then(None)
        .otherwise(
            raw_timestamp.cast(pl.String).str.to_datetime(strict=False, time_zone="UTC")
        )
        .alias("timestamp")
    )
    transaction_hash = pl.coalesce(
        [
            _col_or_null("transaction_hash", schema_names),
            _col_or_null("hash", schema_names),
        ]
    ).cast(pl.String)
    operation_id = _col_or_null("id", schema_names).cast(pl.Int64, strict=False)

    return (
        frame.with_columns(
            [
                ledger_sequence,
                operation_id.alias("operation_id"),
                transaction_hash.alias("transaction_hash"),
                op_type.alias("type"),
                sender.alias("sender"),
                receiver.alias("receiver"),
                asset,
                amount,
                timestamp,
            ]
        )
        .select(
            [
                "ledger_sequence",
                "operation_id",
                "transaction_hash",
                "type",
                "sender",
                "receiver",
                "asset",
                "amount",
                "timestamp",
            ]
        )
        .filter(
            pl.col("ledger_sequence").is_not_null()
            & pl.col("sender").is_not_null()
            & pl.col("transaction_hash").is_not_null()
            & pl.col("timestamp").is_not_null()
        )
        .unique(subset=["transaction_hash", "operation_id"], keep="first")
        .sort(["ledger_sequence", "operation_id"], descending=False)
    )


def preprocess_to_parquet(
    input_path: str | Path,
    output_path: str | Path,
    input_format: BackfillFormat | None = None,
) -> Path:
    """Read a backfill dataset, normalize it, and write Parquet output."""
    frame = scan_backfill_dataset(input_path=input_path, input_format=input_format)
    processed = preprocess_ledger_backfill(frame)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    processed.sink_parquet(str(out), compression="zstd")
    return out
