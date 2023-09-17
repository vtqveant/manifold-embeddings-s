import os
from typing import List, Optional, Any

import torch
from fastapi import FastAPI, APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE
from transformers import AutoTokenizer, AutoModel

MODULE_NAME = "manifold-embeddings-s"
MODEL_NAME = 'rubert-tiny2'
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models', MODEL_NAME)


class Embedding(BaseModel):
    index: int
    object: str = "embedding"
    embedding: List[float]


class RequestDTO(BaseModel):
    input: str


class ResponseDTO(BaseModel):
    object: str = "list"
    data: List[Embedding]
    model: str
    usage: Optional[Any] = None


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)

app = FastAPI(title=MODULE_NAME, version="0.1.0")
prefix_router = APIRouter(prefix="/v1")


@prefix_router.post('/embeddings/')
async def predict(request: RequestDTO) -> ResponseDTO | PlainTextResponse:
    if len(request.input) == 0:
        return ResponseDTO(module_name=MODULE_NAME, task_id=request.task_id, result={})
    if len(request.input) > 1:
        return PlainTextResponse(status_code=HTTP_400_BAD_REQUEST, content="Multiple inputs not supported")

    t = tokenizer(request.input, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    result = embeddings[0].cpu().numpy()

    return ResponseDTO(data=[Embedding(index=0, embedding=result)], model=MODEL_NAME)


@prefix_router.get('/readyz')
async def readyz():
    if model is None:
        return PlainTextResponse(status_code=HTTP_503_SERVICE_UNAVAILABLE)
    else:
        return PlainTextResponse(status_code=HTTP_200_OK)


@prefix_router.get('/healthz')
async def healthz():
    return PlainTextResponse(status_code=HTTP_200_OK)


app.include_router(prefix_router)
