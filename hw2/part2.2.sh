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

module load mujoco gcc arrow
source /project/60004/benhoangquan/robot_learning_2026/.venv/bin/activate

export PYTHONPATH=$PYTHONPATH:/project/60004/benhoangquan/robot_learning_2026/LIBERO
export HF_DATASETS_CACHE=/scratch/benhoangquan/.cache/huggingface/datasets
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

python hw2/dreamer_model_trainer.py \
    model_type=simple \
    planner.horizon=10 \
    planner.num_samples=50 \
    planner.num_elites=5 \
    load_policy=/project/60004/benhoangquan/robot_learning_2026/outputs/2026-03-05/14-47-27/policy.pth \
    experiment.name=q2_policy_cem \
    use_policy=true