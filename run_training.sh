#!/bin/bash

# JAX/CUDA configuration for Nvidia Blackwell
#export JAX_PLATFORMS=cuda
#export JAX_ENABLE_X64=false
#export JAX_DISABLE_JIT=false

# Disable problematic HLO rematerialization that causes memory and timing issues
export XLA_FLAGS="--xla_disable_hlo_passes=hlo-rematerialization"

#environments=( "humanoid" "ant_big_maze" "arm_push_easy" )
#network_depths=( 4 8 16)
#seeds=( 1000 2000 3000 4000 5000)

environments=( "humanoid" )
network_depths=( 4 )
seeds=( 1000 2000 3000 4000 5000)

for env in ${environments[@]}; do
  for depth in ${network_depths[@]}; do
    for seed in ${seeds[@]}; do
        uv run train.py \
          --env_id "$env" \
          --eval_env_id "$env" \
          --num_envs 512 \
          --num_epochs 100 \
          --total_env_steps 100000000 \
          --actor_depth $depth \
          --critic_depth $depth \
          --actor_skip_connections 4 \
          --critic_skip_connections 4 \
          --batch_size 256 \
          --vis_length 1000 \
          --save_buffer 0 \
          --seed $seed \
          --wandb_project_name "rl_with_lejepa" \
          --wandb_entity "k00627350-johannes-kepler-universit-t-linz" \
          --wandb_mode "online"
      done
    done
  done