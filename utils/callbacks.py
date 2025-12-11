import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import TrainerCallback
from .visualization import show_confusion_matrix


class MetricsCallback(TrainerCallback):
    def __init__(self, trainer, cf_dir):
        self.trainer = trainer   # keep a reference
        self.cf_dir = cf_dir
        self.cf_dir_img = os.path.join(cf_dir, "images")
        self.cf_dir_val = os.path.join(cf_dir, "values")

        os.makedirs(cf_dir, exist_ok=True)
        os.makedirs(self.cf_dir_img, exist_ok=True)
        os.makedirs(self.cf_dir_val, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Nothing to do if lists are empty
        if not self.trainer.training_metrics:
            return

        # Concatenate stored batch outputs from the epoch
        new_metrics = {metric: [float(lst[metric]) for lst in self.trainer.training_metrics] for metric in self.trainer.training_metrics[0].keys()}
        
        training_log = {"train_loss": float(np.mean(self.trainer.training_losses))}
        
        # Log mean value of batches for epoch
        for key, val in new_metrics.items():
            training_log[f"train_{key}"] = float(np.mean(val))

        # Create confusion matrix
        show_confusion_matrix(
            saving_loc=os.path.join(self.cf_dir_img, f"confusion_matrix_ep_{int(state.epoch - 1)}.jpg"),
            conf_mat=self.trainer.confmat,
            class_labels=['Background', 'Landslide'],
            )
        pd.DataFrame(self.trainer.confmat, index=[0,1], columns=[0,1]).to_csv(os.path.join(self.cf_dir_val, f"confusion_matrix_ep_{state.epoch - 1}.csv"), sep=';')

        # Clear buffers for next epoch
        self.trainer.training_metrics.clear()
        self.trainer.confmat = np.zeros((2,2), dtype=np.uint64)

        # Log training metric (it will appear in trainer_state.json)
        self.trainer.log(training_log)


class SaveBestPredictionsCallback(TrainerCallback):
    def __init__(self, trainer, save_dir):
        self.trainer = trainer
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = None   # track manually

    @staticmethod
    def save_tif_from_array(src, filename, arr):
        src_images = os.path.join(src, "images")
        src_masks = os.path.join(src, "masks")
        os.makedirs(src_images, exist_ok=True)
        os.makedirs(src_masks, exist_ok=True)

        src_image_file = os.path.join(src_images, filename)
        src_mask_file = os.path.join(src_masks, filename)

        # Saving binary mask
        pil_mask = Image.fromarray(arr.astype(np.uint8))
        pil_mask.save(src_mask_file)

        # Saving rgb versino of mask
        rgb_mask = np.zeros((arr.shape[0], arr.shape[1], 3))
        rgb_mask[arr == 1] = 255
        pil_rgb_mask = Image.fromarray(rgb_mask.astype(np.uint8))
        pil_rgb_mask.save(src_image_file)

    def on_evaluate(self, args, state, control, **kwargs):
        # Save evaluation predictions if best epoch and then clear it
        if state.best_metric == None or state.stateful_callbacks['TrainerControl']['args']['should_save']:
            for filename, preds in self.trainer.eval_preds.items():
                self.save_tif_from_array(self.save_dir, filename, preds)
        self.trainer.eval_preds.clear()


class SavesCurrentStateCallback(TrainerCallback):
    def __init__(self, last_checkpoint_dir, trainer):
        self.trainer = trainer
        self.checkpoint_dir = last_checkpoint_dir

    def on_evaluate(self, args, state, control, **kwargs):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.trainer.save_model(self.checkpoint_dir, _internal_call=True)
        self.trainer.state.save_to_json(os.path.join(self.checkpoint_dir, 'trainer_state.json'))
        self.trainer._save_rng_state(self.checkpoint_dir)
        torch.save(self.trainer.optimizer.state_dict(), os.path.join(self.checkpoint_dir, "optimizer.pt"))
        torch.save(self.trainer.accelerator.scaler.state_dict(), os.path.join(self.checkpoint_dir, "scaler.pt"))
        torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(self.checkpoint_dir, "scheduler.pt"))

