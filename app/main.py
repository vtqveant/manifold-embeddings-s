import os
from typing import List, Optional, Any, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI, APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE
from transformers import AutoTokenizer, AutoModel

VERSION = "0.1.0"

MODEL_NAME = 'all-MiniLM-L6-v2'
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


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)

app = FastAPI(version=VERSION)
prefix_router = APIRouter(prefix="/v1")


@prefix_router.post('/embeddings/', response_model=ResponseDTO)
async def embeddings(request: RequestDTO):
    if len(request.input) == 0:
        return PlainTextResponse(status_code=HTTP_400_BAD_REQUEST, content="No input")
    elif isinstance(request.input, str):
        entries = [request.input]
    else:
        entries = request.input

    encoded_input = tokenizer(entries, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    results = [Embedding(index=idx, embedding=result) for idx, result in enumerate(sentence_embeddings)]
    return ResponseDTO(data=results, model=MODEL_NAME)


@prefix_router.get('/readyz')
async def readyz():
    if model is None:
        return PlainTextResponse(status_code=HTTP_503_SERVICE_UNAVAILABLE)
    else:
        return PlainTextResponse(status_code=HTTP_200_OK)


@prefix_router.get('/livez')
async def healthz():
    return PlainTextResponse(status_code=HTTP_200_OK)


app.include_router(prefix_router)
