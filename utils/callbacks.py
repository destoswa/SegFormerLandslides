import numpy as np
from transformers import TrainerCallback


class PrinterCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer   # keep a reference

    def on_epoch_end(self, args, state, control, **kwargs):
        # Nothing to do if lists are empty
        if not self.trainer.training_metrics:
            return

        # Concatenate stored batch outputs from the epoch
        new_metrics = {metric: [float(lst[metric]) for lst in self.trainer.training_metrics] for metric in self.trainer.training_metrics[0].keys()}
        # mean_iou = float(np.mean(new_metrics['mean_iou']))
        # iou_class_0 = float(np.mean(new_metrics['iou_class_0']))
        # iou_class_1 = float(np.mean(new_metrics['iou_class_1']))
        # loss = float(np.mean(self.trainer.training_losses))
        
        training_log = {"train_loss": float(np.mean(self.trainer.training_losses))}
        
        # Log mean value of batches for epoch
        for key, val in new_metrics.items():
            training_log[f"train_{key}"] = float(np.mean(val))

        # Clear buffers for next epoch
        self.trainer.training_metrics.clear()

        # Log training metric (it will appear in trainer_state.json)
        # self.trainer.log({"train_loss": loss, "train_mean_iou": mean_iou, "train_iou_class_0": iou_class_0, "train_iou_class_1": iou_class_1})
        self.trainer.log(training_log)