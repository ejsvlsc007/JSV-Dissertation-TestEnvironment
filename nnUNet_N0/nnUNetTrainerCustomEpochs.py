import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerCustomEpochs(nnUNetTrainer):
    """
    Custom nnUNet trainer that reads max_num_epochs from the
    NNUNET_NUM_EPOCHS environment variable.
    Falls back to 1000 if not set.
    """

    def __init__(self, plans, configuration, fold, dataset_json, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, **kwargs)
        self.num_epochs = int(os.environ.get("NNUNET_NUM_EPOCHS", 1000))
        self.print_to_log_file(
            f"nnUNetTrainerCustomEpochs: max_num_epochs = {self.num_epochs}"
        )
