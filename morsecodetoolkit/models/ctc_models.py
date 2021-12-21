
import os
from typing import Optional, Dict, Union
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.core import Dataset


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
