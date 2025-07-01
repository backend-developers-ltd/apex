# Stage 1 — Download model with huggingface-hub
FROM python:3.10-slim AS model_downloader

ARG VLLM_MODEL_ID
ARG HF_MODEL_PATH

# Install huggingface-hub
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir huggingface-hub

# Set environment variables
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Copy model downloader script
COPY download_model.py .

# Run download script
RUN python download_model.py --model-name "$VLLM_MODEL_ID" --model-path "$HF_MODEL_PATH"


# Stage 2 — GPU Container
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Declare ARGs again to access build-time values
ARG MODEL_ID
ARG VLLM_MODEL_ID
ARG VLLM_GPU_UTILIZATION
ARG HF_MODEL_PATH

# Promote them to ENV so your app can access them at runtime
ENV MODEL_ID=${MODEL_ID}
ENV VLLM_MODEL_ID=${VLLM_MODEL_ID}
ENV VLLM_GPU_UTILIZATION=${VLLM_GPU_UTILIZATION}
ENV HF_MODEL_PATH=${HF_MODEL_PATH}

WORKDIR /app

# Setup Python (unchanged from your original Dockerfile)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3.9-dev python3.9-distutils curl && \
    # Install pip for python3.9
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    # Make python3 point to python3.9
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy downloaded model
COPY --from=model_downloader ${HF_MODEL_PATH} ${HF_MODEL_PATH}

# Copy app files and install requirements
COPY --from=project_context /gpu_container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=project_context /gpu_container/ ./gpu_container/

# Command to run the application
CMD ["uvicorn", "gpu_container.app:app", "--host", "0.0.0.0", "--port", "8000"]
