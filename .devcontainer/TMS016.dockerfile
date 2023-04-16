FROM python:3.10.10-slim-bullseye

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y gcc && \
    apt-get install -y libsuitesparse-dev && \
    apt-get install -y python3-dev && \
    pip install --upgrade pip