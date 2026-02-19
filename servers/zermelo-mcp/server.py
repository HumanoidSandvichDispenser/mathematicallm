"""MCP server for game tree analysis operations."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gametree-mcp")

# Session store for game tree objects
_store: dict[str, object] = {}
_counter = 0


def _new_id(prefix: str) -> str:
    """Generate a new unique ID for storing objects."""
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


# TODO: Add MCP tool implementations


if __name__ == "__main__":
    mcp.run()
