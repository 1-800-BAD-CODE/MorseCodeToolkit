
import os
from typing import Optional, Dict, Union

from nemo.core import Dataset
from nemo.utils import logging
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.data import audio_to_text_dataset
from omegaconf import DictConfig, open_dict, Container
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch.utils.data import DataLoader


class MorseEncDecCTCModel(EncDecCTCModel):
    """CTC-based encoder-decoder model for morse audio signal decoding.

    CTC-based encoder-decoder model for morse audio signal decoding. Essentially an ASR model that has been
    repurposed.

    """

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if "shuffle" not in train_data_config:
            train_data_config["shuffle"] = True
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    def _setup_dataloader_from_config(self, config: Optional[Dict]) -> DataLoader:
        dataset: Dataset = MorseEncDecCTCModel.from_config_dict(
            config["dataset_params"]
        )
        dataloader: DataLoader = DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=config.get("shuffle", False),
            num_workers=config.get("num_workers", 4),
            collate_fn=dataset.collate_fn
        )
        return dataloader

    def on_validation_epoch_start(self) -> None:
        # Synthetic datasets should reset the state of their RNG now
        ds = self._validation_dl.dataset
        if hasattr(ds, "on_validation_epoch_start") and callable(ds.on_validation_epoch_start):
            ds.on_validation_epoch_start()

    def _setup_transcribe_dataloader(self, config: Dict) -> DataLoader:
        """Creates data loader for inference.

        Creates a dataset compatible with the parent class, which is the class that implements the
        ``transcribe`` method used by this model.

        Args:
            config: Dictionary containing the path to the temp dir and batch size.

        Returns:
            Data loader for an audio-to-char dataset that loads the config's manifest.
        """

        dataset_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.cfg.preprocessor.sample_rate,
            'labels': self.decoder.vocabulary,
        }
        dataset = audio_to_text_dataset.get_char_dataset(config=dataset_config)
        return DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn
        )

    @rank_zero_only
    def maybe_init_from_pretrained_checkpoint(self, cfg: Container, map_location: str = 'cpu'):
        """Overrides ``nemo.core.classes.ModelPT.maybe_init_from_pretrained_checkpoint``.

         Differences from inherited implementation:
             * Clean up a couple things (e.g., allow multiple keys as long as only one is non-null).
             * Support changing vocabulary
             * Do not accept the keyword ``init_from_pretrained_model`` since this project does not store models in the
             cloud.

         Args:
             cfg: Configuration which may optionally specify one of ['init_from_nemo_model', 'init_from_ptl_ckpt'] whose
             values are a filepath to either a ``.nemo`` model or ``.ckpt`` checkpoint, respectively.
             map_location: Where to map the restored model.

         Raises:
             ValueError if more than one of the acceptable keys are non-null.

        """
        # Search config for model to load.
        accepted_keys = ['init_from_nemo_model', 'init_from_ptl_ckpt']
        matched_keys = [key for key in accepted_keys if cfg.get(key, None) is not None]
        num_matches = len(matched_keys)

        # No model to start from.
        if num_matches == 0:
            return

        # More than one would be ambiguous.
        if num_matches > 1:
            raise ValueError(f"Expected at most one model to restore; got {[matched_keys]}")

        # We have exactly one model from which to warm start. Restore it.
        restored_model: MorseEncDecCTCModel
        with open_dict(cfg):
            if cfg.get('init_from_nemo_model', None) is not None:
                # Restore model
                model_path = cfg.pop('init_from_nemo_model')
                restored_model = MorseEncDecCTCModel.restore_from(
                    model_path, map_location=torch.device(map_location), strict=True
                )
            elif cfg.get('init_from_ptl_ckpt', None) is not None:
                # Restore checkpoint
                ckpt_path = cfg.pop('init_from_ptl_ckpt')
                restored_model = MorseEncDecCTCModel.load_from_checkpoint(ckpt_path, map_location=map_location)
            else:
                raise ValueError("Check that all possible keys are accounted for when loading pretrained model!")

        # Change the vocabulary of loaded model, if needed (method does nothing if vocab is same).
        restored_model.change_vocabulary(self.decoder.vocabulary)

        # Restore checkpoint into current model
        self.load_state_dict(restored_model.state_dict(), strict=False)
        del restored_model

        logging.info(f"Model weights loaded from '{model_path}'")
