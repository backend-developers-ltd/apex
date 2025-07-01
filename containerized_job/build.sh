#!/bin/bash

MODEL_ID="WhereIsAI/UAE-Large-V1"
VLLM_MODEL_ID="mrfakename/mistral-small-3.1-24b-instruct-2503-hf"
VLLM_GPU_UTILIZATION=0.8

IMAGE_NAME="backenddevelopersltd/ch-sn1-job"
HF_MODEL_PATH="./hf_model"

DOCKER_BUILDKIT=1 docker build \
    --build-arg MODEL_ID="$MODEL_ID" \
    --build-arg VLLM_MODEL_ID="$VLLM_MODEL_ID" \
    --build-arg HF_MODEL_PATH="$HF_MODEL_PATH" \
    --build-arg VLLM_GPU_UTILIZATION="$VLLM_GPU_UTILIZATION" \
    -t "$IMAGE_NAME" \
    --build-context project_context=../ \
    .
