#!/bin/bash
# Example recaptioning script using open-source models via custom endpoints
# Compatible with: vLLM, Ollama, LM Studio, or any OpenAI-compatible API

# Change to repository root directory
cd "$(dirname "$0")/.." || exit 1

# Add repository root to PYTHONPATH so src module can be imported
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# ==============================================================================
# API Configuration
# ==============================================================================

# Example 1: vLLM server (recommended for production)
# Start vLLM server first:
# python -m vllm.entrypoints.openai.api_server --model llava-hf/llava-v1.6-mistral-7b-hf --port 8000
BASE_URL="http://localhost:8000/v1"
MODEL="llava-hf/llava-v1.6-mistral-7b-hf"

# Example 2: Ollama (easiest to set up)
# Start Ollama first:
# ollama run llama3.2-vision
# BASE_URL="http://localhost:11434/v1"
# MODEL="llama3.2-vision"

# Example 3: LM Studio (GUI-based)
# BASE_URL="http://localhost:1234/v1"
# MODEL="your-model-name"

# Example 4: Remote server
# BASE_URL="http://your-server:8000/v1"
# MODEL="your-model-name"

# Note: API key not required for most open-source endpoints
# If your endpoint requires authentication, set:
# export OPENAI_API_KEY="your-key"

# ==============================================================================
# Dataset Configuration
# ==============================================================================
DATA_DIR="./data"
OUTPUT_DIR="./recaptions_opensource"
NUM_CAPTIONS=5
NUM_CLASSES=5  # Smaller for testing open-source models
NUM_SAMPLES=50  # Smaller for testing
SEED=42

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Build base command with custom endpoint
BASE_CMD="python scripts/recaptioning.py"
COMMON_ARGS="--data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --model ${MODEL} --num_captions ${NUM_CAPTIONS} --num_classes ${NUM_CLASSES} --num_samples ${NUM_SAMPLES} --seed ${SEED} --base_url ${BASE_URL}"

# Recaption CIFAR-100 (pilot: 5 classes, 50 samples each)
echo "========================================="
echo "Recaptioning CIFAR-100 with ${MODEL}..."
echo "Using endpoint: ${BASE_URL}"
echo "========================================="
${BASE_CMD} --dataset cifar100 ${COMMON_ARGS}

# Recaption ImageNet-R (pilot: 5 classes, 50 samples each)
echo "========================================="
echo "Recaptioning ImageNet-R with ${MODEL}..."
echo "========================================="
${BASE_CMD} --dataset imagenet-r ${COMMON_ARGS}

# Recaption ImageNet-100 (pilot: 5 classes, 50 samples each)
echo "========================================="
echo "Recaptioning ImageNet-100 with ${MODEL}..."
echo "========================================="
${BASE_CMD} --dataset imagenet100 ${COMMON_ARGS}

# Recaption Stanford Cars (pilot: 5 classes, 50 samples each)
echo "========================================="
echo "Recaptioning Stanford Cars with ${MODEL}..."
echo "========================================="
${BASE_CMD} --dataset stanford_cars ${COMMON_ARGS}

# Recaption FGVC Aircraft (pilot: 5 classes, 50 samples each)
echo "========================================="
echo "Recaptioning FGVC Aircraft with ${MODEL}..."
echo "========================================="
${BASE_CMD} --dataset fgvc_aircraft ${COMMON_ARGS}

# Recaption CUB-200 (pilot: 5 classes, 50 samples each)
echo "========================================="
echo "Recaptioning CUB-200 with ${MODEL}..."
echo "========================================="
${BASE_CMD} --dataset cub200 ${COMMON_ARGS}

echo "========================================="
echo "All datasets recaptioned with open-source model!"
echo "Results saved in: ${OUTPUT_DIR}"
echo "========================================="
