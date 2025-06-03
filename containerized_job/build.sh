#!/bin/bash

IMAGE_NAME="backenddevelopersltd/ch-sn1-job"
MODEL_NAME="mrfakename/mistral-small-3.1-24b-instruct-2503-hf"

DOCKER_BUILDKIT=1 docker build \
    --build-arg LLM_MODEL="$MODEL_NAME" \
    -t "$IMAGE_NAME" \
    --build-context external_context=../prompting/llms \
    .
