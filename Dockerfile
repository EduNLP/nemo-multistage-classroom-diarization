# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION=3.5"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libboost-all-dev \
        libbz2-dev \
        liblzma-dev \
        zlib1g-dev \
        ffmpeg \
        sox \
        libsox-dev \
        libsndfile1 \
        libsndfile1-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./

RUN set -eux; \
    pip install --upgrade pip; \
    pip install --no-cache-dir "cmake==3.22.6"; \
    tmpdir="$(mktemp -d)"; \
    sed -i '/kenlm/d' requirements.txt; \
    pip install --no-cache-dir -r requirements.txt; \
    git clone https://github.com/kpu/kenlm.git "${tmpdir}/kenlm"; \
    cd "${tmpdir}/kenlm"; \
    git checkout f6c947dc943859e265fabce886232205d0fb2b37; \
    sed -i '1s/.*/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt; \
    pip install --no-cache-dir .; \
    rm -rf "${tmpdir}"

COPY . .

ENV PYTHONPATH=/workspace

RUN mkdir -p logs diarization_output vad_output_frames whisper_output_frames

ENTRYPOINT ["python", "scripts/run_diar_experiment.py"]
CMD ["--help"]