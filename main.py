import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf


logger=logging.getLogger(__name__)
OmegaConf.register_new_resolver("calc", eval, replace=True)


@hydra.main(config_path="configs", config_name="base", version_base="1.1")
def main(cfg: DictConfig) -> None:
    OmegaConf.save(cfg, "hparams.yaml")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    reinl = hydra.utils.instantiate(
        cfg.pipeline,
        model_suite=cfg.model,
        reward=cfg.reward,
        logger=cfg.logger,
    )
    reinl.run_rl()


if __name__ == '__main__':
    main()
