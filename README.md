# Mathematicallm Monorepo

A monorepo of mathematical reasoning tools exposed via MCP servers for use by LLMs.

## Structure

```
mathematicallm/
├── pyproject.toml                  # uv workspace root
├── README.md
│
├── modules/
│   ├── graph/                      # Graph theory library
│   │   ├── pyproject.toml
│   │   ├── src/mathematicallm_graph/
│   │   └── tests/
│   │
│   └── gametree/                   # Game tree analysis library
│       ├── pyproject.toml
│       ├── src/mathematicallm_gametree/
│       └── tests/
│
└── servers/
    ├── graph-mcp/                  # MCP server for graph operations
    │   ├── pyproject.toml
    │   └── server.py
    │
    └── gametree-mcp/               # MCP server for game tree analysis
        ├── pyproject.toml
        └── server.py
```

## Architecture

### Modules
- **Location**: `modules/<name>/`
- **Package name**: `mathematicallm-<name>`
- **Import name**: `mathematicallm_<name>`
- **Dependencies**: External libraries + other workspace modules only
- **Requirement**: Must be usable as standalone Python libraries

### MCP Servers
- **Location**: `servers/<name>-mcp/`
- **Purpose**: Thin wrappers around their corresponding module
- **Dependencies**: Only their corresponding module + `mcp[cli]`
- **Pattern**: Maintain session store with generated IDs for object references

## Dependency Rules

- ✅ modules → external libraries, other modules
- ✅ servers → their own module, mcp[cli]
- ❌ servers → other modules (except their own)
- ❌ servers → other servers
- ❌ modules → mcp

## Adding a New Module + Server

1. Create `modules/<name>/pyproject.toml` with library dependencies only
2. Create `modules/<name>/src/mathematicallm_<name>/__init__.py`
3. Create `modules/<name>/tests/`
4. Add `"modules/<name>"` to workspace members in root `pyproject.toml`
5. Create `servers/<name>-mcp/pyproject.toml` depending on `mathematicallm-<name>` + `mcp[cli]`
6. Create `servers/<name>-mcp/server.py` as thin MCP wrapper
7. Add `"servers/<name>-mcp"` to workspace members in root `pyproject.toml`

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and workspace configuration.

```bash
# Install dependencies
uv sync

# Run a specific MCP server
uv run --directory servers/graph-mcp python server.py

# Run tests for a module
uv run --directory modules/graph pytest
```
