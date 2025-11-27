import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
import numpy as np
from time import time

from utils.dataset import SegmentationDataset
from utils.trainer import TrainValMetricsTrainer

def training_model():

    time_start = time()

    # ----------------------------------------------------------
    # 2) Load model + processor
    # ----------------------------------------------------------
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True  # <- Important for custom classes
    )

    # Set number of classes = your mask max label + 1
    num_classes = 2  # <-- change this to your dataset!
    model.decode_head.classifier = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.config.num_labels = num_classes


    # ----------------------------------------------------------
    # 3) Create datasets + dataloaders
    # ----------------------------------------------------------
    train_ds = SegmentationDataset(
        image_dir="dataset/images",
        mask_dir="dataset/masks",
        processor=processor
    )

    # print("training set size: ", len(train_ds))

    # OPTIONAL: Split train/val
    val_split = 0.1
    num_epochs = 5
    batch_size = 8
    split_idx = int(len(train_ds) * (1 - val_split))
    train_subset, val_subset = torch.utils.data.random_split(train_ds, [split_idx, len(train_ds) - split_idx])


    # ----------------------------------------------------------
    # 4) Training arguments
    # ----------------------------------------------------------

    training_args = TrainingArguments(
        output_dir="./segformer_output_2",  # Where checkpoints and logs are saved
        num_train_epochs=num_epochs,        # Total number of epochs
        per_device_train_batch_size=batch_size,      # Adjust according to your GPU memory
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,                  # Regularization

        # Logging
        logging_dir="./logs",               # TensorBoard logs
        logging_strategy="steps",           # Log every N steps
        logging_steps=len(train_subset),    # Adjust for dataset size
        log_level="info",
        report_to=["tensorboard"],          # Can add "wandb" if desired

        # Checkpoints
        save_strategy="epoch",              # Save checkpoint at the end of each epoch
        save_total_limit=3,                 # Keep last 3 checkpoints
        save_steps=None,                    # Not used when saving by epoch

        # Evaluation
        eval_strategy="epoch",              # Evaluate at the end of each epoch
        load_best_model_at_end=True,        # Load checkpoint with best metric
        metric_for_best_model="mean_iou",   # Adjust if using other metrics

        # Others
        fp16=True,                          # Mixed precision (if GPU supports)
        gradient_accumulation_steps=1,      # Increase effective batch size if needed
        dataloader_num_workers=4,           # Adjust according to CPU cores
        disable_tqdm=False
    )

    # ----------------------------------------------------------
    # 5) HuggingFace Trainer
    # ----------------------------------------------------------
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     preds = np.argmax(logits, axis=1)
    #     acc = (preds == labels).mean()  # simple pixel accuracy

    #     lst_accs = []
    #     for val in range(num_classes):
    #         mask = labels == val
    #         lst_accs.append((preds[mask] == labels[mask]).mean())
    #     acc_mean = np.mean(lst_accs)
    #     return {"pixel_accuracy": acc, "pixel_accuracy_mean": acc_mean}
    
    def compute_metrics(p):
        """
        Compute per-class IoU and mean IoU for semantic segmentation predictions.

        p.predictions: (batch_size, num_classes, H_pred, W_pred)
        p.label_ids:   (batch_size, H_lbl, W_lbl)
        """
        preds = p.predictions  # raw logits
        labels = p.label_ids   # ground-truth labels

        # Resize predictions to match label size
        logits = torch.nn.functional.interpolate(
            torch.tensor(preds),
            size=labels.shape[-2:],   # (H_lbl, W_lbl)
            mode="bilinear",
            align_corners=False
        )

        # Final predicted class mask
        pred = logits.argmax(dim=1).cpu().numpy()  # (batch_size, H_lbl, W_lbl)

        num_classes = preds.shape[1]
        ious = []

        for cls in range(num_classes):
            pred_cls = (pred == cls)
            label_cls = (labels == cls)

            intersection = np.logical_and(pred_cls, label_cls).sum()
            union = np.logical_or(pred_cls, label_cls).sum()

            iou = np.nan if union == 0 else intersection / union
            ious.append(iou)

        ious = np.array(ious)
        mean_iou = np.nanmean(ious)

        metrics = {"mean_iou": mean_iou}
        for cls, iou in enumerate(ious):
            metrics[f"iou_class_{cls}"] = iou

        return metrics


    # trainer = Trainer(
    trainer = TrainValMetricsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        # eval_dataset={"val": val_subset, "train": train_subset},
        eval_dataset= val_subset,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )


    # ----------------------------------------------------------
    # 6) Train
    # ----------------------------------------------------------
    trainer.train()

    # ----------------------------------------------------------
    # 7) Save final model
    # ----------------------------------------------------------
    trainer.save_model("./segformer_trained_model")
    processor.save_pretrained("./segformer_trained_model")

    # ----------------------------------------------------------
    # 7) Save final model
    # ----------------------------------------------------------
    # visualization(trainer)


    # Show duration of process
    delta_time_loop = time() - time_start
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"---\n\n==== Training completed in {hours}:{min}:{sec} ====\n")

