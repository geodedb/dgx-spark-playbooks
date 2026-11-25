import asyncio
import sys

from geode_client import Client, TransportType


async def main():
    client = Client(
        host="127.0.0.1",
        port=3161,
        username="geode",
        password="geodesic",
        skip_verify=True,
        transport_type=TransportType.QUIC,
    )
    async with client.connection() as conn:
        page, _ = await conn.query("MATCH (c:DocumentChunk) RETURN count(c) AS n")
        chunks = list(page.rows)[0]["n"].as_int
        page, _ = await conn.query("MATCH (p:ProcessedDocument) RETURN count(p) AS n")
        files = list(page.rows)[0]["n"].as_int
    print(f"{chunks} {files}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"query-failed: {e}", file=sys.stderr)
        sys.exit(1)
