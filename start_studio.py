"""
Harsh AI Video Studio - Universal Server Launcher.
Starts FastAPI backend, UI, and async database on 0.0.0.0:8000.
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import uvicorn
from app.main import app

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Harsh AI Video Studio on http://0.0.0.0:8000")
    print("Swagger API Docs: http://0.0.0.0:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
