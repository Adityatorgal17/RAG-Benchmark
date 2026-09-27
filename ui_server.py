from __future__ import annotations

import sys
import os

# Add src to path for module discovery
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ragc.ui_backend import build_ui_response, get_ui_runtime_status

app = FastAPI(title="RAG Explorer")


class RunRequest(BaseModel):
    system: str
    question: str


app.mount("/static", StaticFiles(directory="ui"), name="static")


@app.get("/")
def index():
    return FileResponse("ui/index.html")


@app.post("/api/run")
def run_system(req: RunRequest):
    return build_ui_response(req.system, req.question)


@app.get("/api/status")
def runtime_status():
    return get_ui_runtime_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)