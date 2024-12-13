#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir $SCRIPT_DIR/../checkpoints
wget -P $SCRIPT_DIR/../checkpoints/ -N https://huggingface.co/yeates/PromptFix/resolve/main/promptfix.ckpt

echo "Pre-trained model downloaded to checkpoints/promptfix.ckpt"
echo "Begin inference..."
CUDA_VISIBLE_DEVICES='5, 6, 7' python $SCRIPT_DIR/../inference.py