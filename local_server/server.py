"""
server.py

FastAPI server exposing the local language model.
"""

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
import uvicorn

from local_server.inference import generate_answer
from local_server.config import HOST
from local_server.config import PORT

app = FastAPI(
    title="Local LLM Server",
    version="1.0.0"
)


class PromptRequest(BaseModel):
    prompt: str


@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {
        "status": "ok"
    }


@app.post("/generate")
def generate(request: PromptRequest):
    """
    Generate an answer using the local model.
    """

    try:

        answer = generate_answer(request.prompt)

        return {
            "answer": answer
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


if __name__ == "__main__":

    uvicorn.run(
        app,
        host=HOST,
        port=PORT
    )