#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hoangquan.tran410@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=part1.out
#SBATCH --error=part1.err
#SBATCH --job-name=part1

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python hw2/dreamer_model_trainer.py \
    model_type=simple \
    planner.type=cem \
    planner.horizon=10 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    experiment.name=q1_simple_cem \
    use_policy=false \