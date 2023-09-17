import os
from typing import List, Optional, Any, Union

import torch
from fastapi import FastAPI, APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE
from transformers import AutoTokenizer, AutoModel

VERSION = "0.1.0"

MODULE_NAME = "manifold-embeddings-s"
MODEL_NAME = 'rubert-tiny2'
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models', MODEL_NAME)


class Embedding(BaseModel):
    index: int
    object: str = "embedding"
    embedding: List[float]


class RequestDTO(BaseModel):
    input: Union[str, List[str]]


class ResponseDTO(BaseModel):
    object: str = "list"
    data: List[Embedding]
    model: str
    usage: Optional[Any] = None


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)

app = FastAPI(title=MODULE_NAME, version=VERSION)
prefix_router = APIRouter(prefix="/v1")


@prefix_router.post('/embeddings/', response_model=ResponseDTO)
async def embeddings(request: RequestDTO):
    if len(request.input) == 0:
        return PlainTextResponse(status_code=HTTP_400_BAD_REQUEST, content="No input")
    elif isinstance(request.input, str):
        entries = [request.input]
    else:
        entries = request.input

    results = []
    for idx, entry in enumerate(entries):
        t = tokenizer(request.input, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        result = embeddings[0].cpu().numpy()
        results.append(Embedding(index=idx, embedding=result))

    return ResponseDTO(data=results, model=MODEL_NAME)


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
