# Shim: re-export the real app from my_env.server.app
# This file exists so that `openenv validate` and `uvicorn server.app:app`
# find the expected entry point at the repo root.

from my_env.server.app import app  # noqa: F401


def main() -> None:
    """Entry point for direct execution."""
    from my_env.server.app import main as _main
    _main()


if __name__ == "__main__":
    main()
