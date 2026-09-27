"""
AssureX Backend Server Launcher
===============================
Starts the Uvicorn ASGI server hosting the FastAPI application.
Usage:
    python run_server.py [--port 8000] [--host 0.0.0.0] [--reload]
"""

import sys
import argparse
import uvicorn
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run AssureX FastAPI Backend Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Binding TCP port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")

    args = parser.parse_args()

    print("=" * 65)
    print(" Starting AssureX Claims Processing Engine Backend")
    print(f" Address     : http://{args.host}:{args.port}")
    print(f" Swagger Docs: http://localhost:{args.port}/docs")
    print(f" ReDoc Docs  : http://localhost:{args.port}/redoc")
    print("=" * 65)

    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
