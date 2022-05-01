#!/usr/bin/env bash

declare algo_alias="gem+tbp"
declare algo="GEM"

declare root="$HOME/workspace/GEM/gem_mujoco"
declare max_steps=1000


declare -a gpus=(0 1 2)
declare -a envs=("Hopper-v3" "Walker2d-v3" "Swimmer-v3" "InvertedDoublePendulum-v2" "InvertedPendulum-v2")
declare -a steps=(400001 400001 400001 100001 100001)
declare -a envs_alias=("hopper" "walker" "swimmer" "doublependulum" "pendulum")
export PYTHONPATH=$root
OPENAI_LOGDIR=./log_gem/mujoco/gem+tbp python $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=GEM --comment=hopper_gem+tbp_0 --seed 0 --env-id=Hopper-v3