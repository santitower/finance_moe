#!/bin/bash
# Download all 5 GGUF models for the Finance MoE system
# Total size: ~61 GB. Takes 15-30 minutes depending on connection.

set -e

MODELS_DIR="/Users/fsd/finance_moe/models"
mkdir -p "$MODELS_DIR"

echo "=== Finance MoE Model Downloader ==="
echo ""

# 1. Fin-R1 (7B) — Financial domain specialist
echo "[1/5] Downloading Fin-R1 (7B, ~4.5 GB)..."
huggingface-cli download bartowski/SUFE-AIFLM-Lab_Fin-R1-GGUF \
  --include "SUFE-AIFLM-Lab_Fin-R1-Q4_K_M.gguf" \
  --local-dir "$MODELS_DIR" 2>&1

# 2. Qwen3.6-35B-A3B MoE (35B total / 3B active)
echo "[2/5] Downloading Qwen3.6-35B-A3B MoE (~22.3 GB)..."
huggingface-cli download bartowski/Qwen_Qwen3.6-35B-A3B-GGUF \
  --include "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --local-dir "$MODELS_DIR" 2>&1

# 3. DeepSeek-R1-Distill-Qwen-14B
echo "[3/5] Downloading DeepSeek-R1-Distill-Qwen-14B (~8.5 GB)..."
huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF \
  --include "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf" \
  --local-dir "$MODELS_DIR" 2>&1

# 4. Qwen3.8-27B
echo "[4/5] Downloading Qwen3.8-27B (~16.5 GB)..."
huggingface-cli download bartowski/Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-Q4_K_M.gguf" \
  --local-dir "$MODELS_DIR" 2>&1

# 5. Qwen3-14B-abliterated
echo "[5/5] Downloading Qwen3-14B-abliterated (~9.4 GB)..."
huggingface-cli download bartowski/huihui-ai_Qwen3-14B-abliterated-GGUF \
  --include "huihui-ai_Qwen3-14B-abliterated-Q4_K_M.gguf" \
  --local-dir "$MODELS_DIR" 2>&1

echo ""
echo "=== Download Complete ==="
ls -lh "$MODELS_DIR"/*.gguf
echo ""
echo "NOTE: Actual filenames may differ slightly from config.py expectations."
echo "Check filenames above and update config.py or expert classes if needed."
