FROM python:3.8-slim AS builder
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    git clone --depth=1 --branch=main https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 /root/app/models/all-MiniLM-L6-v2

FROM huggingface/transformers-inference:4.24.0-pt1.13-cpu
COPY --from=builder /root/app/models/all-MiniLM-L6-v2 /root/app/models/all-MiniLM-L6-v2
COPY ./app /root/app/app
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /root/app
ENV MAX_WORKERS="1" \
    WEB_CONCURRENCY="1"

CMD uvicorn --host 0.0.0.0 --port ${PORT:-80} app.main:app