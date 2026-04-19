#!/bin/bash

# JAX/CUDA configuration for Nvidia Blackwell
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=false
export JAX_DISABLE_JIT=false

# Disable problematic HLO rematerialization that causes memory and timing issues
export XLA_FLAGS="--xla_disable_hlo_passes=hlo-rematerialization"

# Memory-optimized training configuration for RTX 4000 (20GB VRAM)
uv run train.py \
  --env_id "humanoid" \
  --eval_env_id "humanoid" \
  --num_epochs 100 \
  --total_env_steps 100000000 \
  --critic_depth 32 \
  --actor_depth 32 \
  --actor_skip_connections 4 \
  --critic_skip_connections 4 \
  --batch_size 256 \
  --vis_length 1000 \
  --save_buffer 0