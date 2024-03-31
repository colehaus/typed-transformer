# This version results in a substantially larger image
# and ghcr.io seems somewhat slower
# FROM ghcr.io/nvidia/jax:nightly-2023-11-03
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \ 
    # Required for `add-apt-repository`
    apt-get install -y --no-install-recommends software-properties-common && \
    # Required for `python3.12`
    add-apt-repository ppa:deadsnakes/ppa -y && \
    # venv required for ensurepip
    apt-get install -y --no-install-recommends python3.12 python3.12-venv
RUN python3.12 -m ensurepip --upgrade

# Not required but improves caching
RUN python3.12 -m pip install --no-cache-dir "jax[cuda12_pip]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN python3.12 -m pip install --no-cache-dir poetry poetry-plugin-export

RUN apt-get install -y --no-install-recommends git htop ncdu
# RUN apt-get install -y --no-install-recommends build-essential cmake python3.12-dev swig

COPY ./pyproject.toml ./poetry.lock /tmp/
COPY ./ml_scratch/  /tmp/ml_scratch/
RUN cd /tmp && \
    poetry export --output requirements.txt --extras "dev gradient gpu" && \
    python3.11 -m pip install --no-cache-dir --requirement requirements.txt && \ 
    rm -r /tmp/*

SHELL ["/bin/bash", "-c"]
EXPOSE 6006 8888
CMD python3.12 -m jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True
