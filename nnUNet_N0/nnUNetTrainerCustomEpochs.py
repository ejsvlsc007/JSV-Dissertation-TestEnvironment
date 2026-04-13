import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerCustomEpochs(nnUNetTrainer):
    """
    Custom nnUNet trainer that reads max_num_epochs from the
    NNUNET_NUM_EPOCHS environment variable, set by the notebook
    in Cell 2 (NUM_EPOCHS). Falls back to 1000 if not set.

    Usage (via CLI):
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerCustomEpochs
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=None):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = int(os.environ.get("NNUNET_NUM_EPOCHS", 1000))
        self.print_to_log_file(
            f"nnUNetTrainerCustomEpochs: max_num_epochs set to {self.num_epochs} "
            f"(from NNUNET_NUM_EPOCHS env var)"
        )
