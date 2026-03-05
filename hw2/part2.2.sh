#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=hoangquan.tran410@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=part2.out
#SBATCH --error=part2.err
#SBATCH --job-name=part2

export PYTHONPATH=$PYTHONPATH:/teamspace/studios/this_studio/LIBERO
python hw2/dreamer_model_trainer.py \
    model_type=simple \
    planner.horizon=10 \
    planner.num_samples=50 \
    planner.num_elites=5 \
    load_policy=/teamspace/studios/this_studio/outputs/2026-03-05/18-30-23/policy.pth \
    experiment.name=q2_policy_cem \
    use_policy=true