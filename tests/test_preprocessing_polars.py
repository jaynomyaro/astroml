from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from astroml.preprocessing.ledger_backfill import preprocess_to_parquet


def _write_ndjson(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_preprocess_to_parquet_normalizes_operations(tmp_path: Path) -> None:
    input_file = tmp_path / "ops.ndjson"
    output_file = tmp_path / "normalized.parquet"
    _write_ndjson(
        input_file,
        [
            {
                "id": "1",
                "ledger": "100",
                "type": "payment",
                "source_account": "G_SENDER",
                "to": "G_RECEIVER",
                "amount": "12.5",
                "asset_type": "native",
                "created_at": "2024-01-15T12:30:00Z",
                "transaction_hash": "tx_a",
            },
            {
                "id": "2",
                "ledger": 101,
                "type": "create_account",
                "source_account": "G_FUNDER",
                "account": "G_NEW",
                "starting_balance": "1000.0000000",
                "asset_type": "native",
                "created_at": "2024-01-15T12:31:00Z",
                "transaction_hash": "tx_b",
            },
            {
                "id": 3,
                "ledger_sequence": 102,
                "type": "change_trust",
                "source_account": "G_TRUSTOR",
                "asset_type": "credit_alphanum4",
                "asset_code": "USDC",
                "asset_issuer": "G_ISSUER",
                "created_at": "2024-01-15T12:32:00Z",
                "transaction_hash": "tx_c",
            },
        ],
    )

    out = preprocess_to_parquet(input_file, output_file)
    assert out == output_file
    assert output_file.exists()

    df = pl.read_parquet(output_file).sort("ledger_sequence")
    assert df.columns == [
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
    assert df.height == 3
    assert df["receiver"].to_list() == ["G_RECEIVER", "G_NEW", None]
    assert df["amount"].to_list() == [12.5, 1000.0, None]
    assert df["asset"].to_list() == ["XLM", "XLM", "USDC:G_ISSUER"]


def test_preprocess_to_parquet_scans_directory(tmp_path: Path) -> None:
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    _write_ndjson(
        in_dir / "part-1.ndjson",
        [
            {
                "id": 10,
                "ledger": 200,
                "type": "payment",
                "source_account": "G_A",
                "to": "G_B",
                "amount": "1",
                "asset_type": "native",
                "created_at": "2024-01-15T12:30:00Z",
                "transaction_hash": "tx_10",
            }
        ],
    )
    _write_ndjson(
        in_dir / "part-2.ndjson",
        [
            {
                "id": 11,
                "ledger": 201,
                "type": "payment",
                "source_account": "G_C",
                "to": "G_D",
                "amount": "2",
                "asset_type": "native",
                "created_at": "2024-01-15T12:31:00Z",
                "transaction_hash": "tx_11",
            }
        ],
    )
    output_file = tmp_path / "out.parquet"

    preprocess_to_parquet(in_dir, output_file)

    df = pl.read_parquet(output_file)
    assert df.height == 2
    assert set(df["ledger_sequence"].to_list()) == {200, 201}
