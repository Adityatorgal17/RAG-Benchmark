from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ragc.ui_backend import build_ui_response

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
