import asyncio
import json
from urllib.parse import parse_qs, urlparse

import pytest

from astroml.ingestion import HorizonStreamError, HorizonStreamingClient


def test_package_exports_horizon_streaming_client():
    """Guard against `astroml.ingestion`'s __init__ dropping these exports again.

    `HorizonStreamingClient`/`HorizonStreamError` are defined in
    `astroml.ingestion.horizon_stream` but must also be importable from the
    `astroml.ingestion` package root, since callers (including this test
    module) import them that way (issue #976).
    """
    import astroml.ingestion as ingestion_pkg
    from astroml.ingestion.horizon_stream import (
        HorizonStreamError as DirectHorizonStreamError,
    )
    from astroml.ingestion.horizon_stream import (
        HorizonStreamingClient as DirectHorizonStreamingClient,
    )

    assert ingestion_pkg.HorizonStreamingClient is DirectHorizonStreamingClient
    assert ingestion_pkg.HorizonStreamError is DirectHorizonStreamError


def test_horizon_stream_ingests_transactions():
    async def run_test():
        received = []
        received_event = asyncio.Event()

        async def handler(reader, writer):
            request_line = await reader.readline()
            assert request_line.startswith(b"GET /transactions?")
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break

            payload = {"id": "tx-1", "paging_token": "12345"}
            response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Connection: close\r\n\r\n"
                f"data: {json.dumps(payload)}\n\n"
            )
            writer.write(response.encode("utf-8"))
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            reconnect_delay=0.5,
            max_reconnect_delay=1.0,
        )

        async def on_transaction(tx):
            received.append(tx)
            received_event.set()

        try:
            await client.start(on_transaction)
            await asyncio.wait_for(received_event.wait(), timeout=1.0)
            await client.stop()
        finally:
            await client.stop()
            server.close()
            await server.wait_closed()

        assert received == [{"id": "tx-1", "paging_token": "12345"}]
        assert client.cursor == "12345"

    asyncio.run(run_test())


def test_horizon_stream_reconnects_automatically():
    async def run_test():
        received_ids = []
        request_cursors = []
        received_event = asyncio.Event()
        connection_count = 0

        async def handler(reader, writer):
            nonlocal connection_count
            connection_count += 1

            request_line = await reader.readline()
            target = request_line.decode("ascii").split(" ")[1]
            query = parse_qs(urlparse(target).query)
            request_cursors.append(query.get("cursor", [""])[0])

            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break

            payload = {"id": f"tx-{connection_count}", "paging_token": str(connection_count)}
            response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Connection: close\r\n\r\n"
                f"data: {json.dumps(payload)}\n\n"
            )
            writer.write(response.encode("utf-8"))
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            reconnect_delay=0.05,
            max_reconnect_delay=0.1,
        )

        async def on_transaction(tx):
            received_ids.append(tx["id"])
            if len(received_ids) >= 2:
                received_event.set()

        try:
            await client.start(on_transaction)
            await asyncio.wait_for(received_event.wait(), timeout=2.0)
            await client.stop()
        finally:
            await client.stop()
            server.close()
            await server.wait_closed()

        assert connection_count >= 2
        assert received_ids[:2] == ["tx-1", "tx-2"]
        assert request_cursors[0] == "now"
        assert request_cursors[1] == "1"

    asyncio.run(run_test())


def test_horizon_stream_graceful_shutdown():
    async def run_test():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break

            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()

            try:
                while True:
                    writer.write(b": keepalive\n\n")
                    await writer.drain()
                    await asyncio.sleep(0.05)
            except Exception:
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            reconnect_delay=0.05,
            max_reconnect_delay=0.1,
        )

        async def on_transaction(_tx):
            return None

        try:
            await client.start(on_transaction)
            await asyncio.sleep(0.2)
            await asyncio.wait_for(client.stop(), timeout=1.0)
        finally:
            await client.stop()
            server.close()
            await server.wait_closed()

        assert client._task is None

    asyncio.run(run_test())


def test_horizon_stream_raises_on_non_200_status():
    """`_consume_stream` surfaces a `HorizonStreamError` for non-200 responses."""

    async def run_test():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break
            writer.write(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(base_url=f"http://127.0.0.1:{port}")

        async def on_transaction(_tx):
            return None

        try:
            with pytest.raises(HorizonStreamError, match="503"):
                await client._consume_stream(on_transaction)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run_test())
