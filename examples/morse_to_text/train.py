
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from morsecodetoolkit.models.ctc_models import MorseEncDecCTCModel


@hydra_runner(config_path="conf", config_name="quartznet_5x3")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    trainer = Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model: MorseEncDecCTCModel = MorseEncDecCTCModel(cfg=cfg.model, trainer=trainer)

    # Warm start if pre-trained model is specified. Currently, resizing any layers will break this function.
    model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(model)


if __name__ == '__main__':
    main()
