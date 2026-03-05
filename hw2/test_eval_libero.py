# Save as: hw2/test_eval_libero_env.py

import torch
from omegaconf import DictConfig
import hydra

from sim_eval import eval_libero

import sys
sys.path.append("/teamspace/studios/this_studio/LIBERO")


@hydra.main(config_path="./conf", config_name="64pix-pose")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We only care about env creation + rendering here, not the model.
    # Passing None is fine because the crash you’re seeing happens
    # when creating the LIBERO DenseRewardEnv, before the model is used.
    # Override eval_tasks to empty so we test benchmark/import loading only,
    # avoiding the model-dependent action loop.
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.sim.eval_tasks = []

    eval_libero(
        model=None,
        device=device,
        cfg=cfg,
        iter_=0,
        log_dir=".",
        tokenizer=None,
        text_model=None,
        wandb=None,
    )
    print("ENV CREATION TEST PASSED: eval_libero completed without errors.")


if __name__ == "__main__":
    main()