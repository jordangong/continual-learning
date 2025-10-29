#!/bin/bash
# Example recaptioning script for pilot study

# Change to repository root directory
cd "$(dirname "$0")/.." || exit 1

# Add repository root to PYTHONPATH so src module can be imported
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# ==============================================================================
# API Configuration - Choose ONE of the following options:
# ==============================================================================

# Option 1: OpenAI API (default)
export OPENAI_API_KEY="your-api-key-here"
BASE_URL=""
MODEL="gpt-4o-mini"

# Option 2: Custom endpoint with open-source model (e.g., vLLM, Ollama)
# Uncomment and configure for open-source models:
# BASE_URL="http://localhost:8000/v1"  # vLLM endpoint
# BASE_URL="http://localhost:11434/v1"  # Ollama endpoint
# MODEL="llama3.2-vision"  # Your model name
# unset OPENAI_API_KEY  # Optional: unset if endpoint doesn't need key

# ==============================================================================
# Dataset Configuration
# ==============================================================================
DATA_DIR="./data"
OUTPUT_DIR="./recaptions"
NUM_CAPTIONS=5
NUM_CLASSES=10
NUM_SAMPLES=100
SEED=42

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Build base command with optional base_url
BASE_CMD="python scripts/recaptioning.py"
COMMON_ARGS="--data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --model ${MODEL} --num_captions ${NUM_CAPTIONS} --num_classes ${NUM_CLASSES} --num_samples ${NUM_SAMPLES} --seed ${SEED}"
if [ -n "${BASE_URL}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --base_url ${BASE_URL}"
fi

# Recaption CIFAR-100 (pilot: 10 classes, 100 samples each)
echo "========================================="
echo "Recaptioning CIFAR-100..."
echo "========================================="
${BASE_CMD} --dataset cifar100 ${COMMON_ARGS}

# Recaption ImageNet-R (pilot: 10 classes, 100 samples each)
echo "========================================="
echo "Recaptioning ImageNet-R..."
echo "========================================="
${BASE_CMD} --dataset imagenet-r ${COMMON_ARGS}

# Recaption ImageNet-100 (pilot: 10 classes, 100 samples each)
echo "========================================="
echo "Recaptioning ImageNet-100 with ${MODEL}..."
echo "========================================="
${BASE_CMD} --dataset imagenet100 ${COMMON_ARGS}

# Recaption Stanford Cars (pilot: 10 classes, 100 samples each)
echo "========================================="
echo "Recaptioning Stanford Cars..."
echo "========================================="
${BASE_CMD} --dataset stanford_cars ${COMMON_ARGS}

# Recaption FGVC Aircraft (pilot: 10 classes, 100 samples each)
echo "========================================="
echo "Recaptioning FGVC Aircraft..."
echo "========================================="
${BASE_CMD} --dataset fgvc_aircraft ${COMMON_ARGS}

# Recaption CUB-200 (pilot: 10 classes, 100 samples each)
echo "========================================="
echo "Recaptioning CUB-200..."
echo "========================================="
${BASE_CMD} --dataset cub200 ${COMMON_ARGS}

echo "========================================="
echo "All datasets recaptioned!"
echo "Results saved in: ${OUTPUT_DIR}"
echo "========================================="
