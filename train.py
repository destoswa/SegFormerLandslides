import os
import json
import torch
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
)
import numpy as np
from time import time
from datetime import datetime
from omegaconf import OmegaConf

from utils.dataset import SegmentationDataset
from utils.trainer import TrainValMetricsTrainer, collate_with_filename
from utils.metrics import compute_metrics
from utils.callbacks import MetricsCallback, SaveBestPredictionsCallback, SavesCurrentStateCallback
from utils.visualization import show_iou_per_class, show_loss_pa, show_mean_iou_dice

# If batch size too big, fails instead of slowing down
torch.cuda.set_per_process_memory_fraction(0.95)


def training_model(args):
    OUTPUT_DIR = args.train.output_dir
    OUTPUT_SUFFIXE = args.train.output_suffixe
    VAL_SPLIT = args.train.val_split
    NUM_EPOCHS = args.train.num_epochs
    NUM_WORKERS = args.train.num_workers
    BATCH_SIZE = args.train.batch_size
    LEARNING_RATE = args.train.learning_rate
    WEIGHT_DECAY = args.train.weight_decay
    PRETRAINED_MODEL = args.train.pretrained_model
    DATASET_DIR = args.dataset.dataset_dir

    RESUME_FROM_EXISTING = args.train.resume_from_existing
    EXISTING_DIR_TO_RESUME_FROM = os.path.join(args.train.existing_dir, 'last_checkpoint') if RESUME_FROM_EXISTING else None

    # Create architecture
    RESULTS_DIR = os.path.join(OUTPUT_DIR, datetime.now().strftime(r"%Y%m%d_%H%M%S_") + f"{NUM_EPOCHS}_epochs_" + OUTPUT_SUFFIXE)
    LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
    CONFMAT_DIR = os.path.join(LOG_DIR, "confmats")
    BESTPREDS_DIR = os.path.join(LOG_DIR, "best_preds")
    IMG_DIR = os.path.join(RESULTS_DIR, 'images')
    LAST_CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'last_checkpoint')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(CONFMAT_DIR, exist_ok=True)

    time_start = time()

    # Load model + processor
    num_classes = 2  # <-- change this to your dataset!

    processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL, use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # <- Important for custom classes
    )

    # Create datasets + dataloaders
    train_ds = SegmentationDataset(
        image_dir=os.path.join(DATASET_DIR, "images"),
        mask_dir=os.path.join(DATASET_DIR, "masks"),
        processor=processor
    )

    # Split train/val
    split_idx = int(len(train_ds) * (1 - VAL_SPLIT))
    train_subset, val_subset = torch.utils.data.random_split(train_ds, [split_idx, len(train_ds) - split_idx])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,  # Where checkpoints and logs are saved
        num_train_epochs=NUM_EPOCHS,        # Total number of epochs
        per_device_train_batch_size=BATCH_SIZE,      # Adjust according to your GPU memory
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,                  # Regularization

        # Logging
        logging_dir=LOG_DIR,                # TensorBoard logs
        logging_strategy="steps",           # Log every N steps
        logging_steps=len(train_subset),    # Adjust for dataset size
        log_level="info",

        # Checkpoints
        save_strategy="epoch",              # Save checkpoint at the end of each epoch
        save_total_limit=3,                 # Keep last 3 checkpoints
        save_steps=None,                    # Not used when saving by epoch

        # Evaluation
        eval_strategy="epoch",              # Evaluate at the end of each epoch
        load_best_model_at_end=True,        # Load checkpoint with best metric
        metric_for_best_model="loss",       # Adjust if using other metrics

        # Others
        fp16=True,                          # Mixed precision (if GPU supports)
        gradient_accumulation_steps=1,      # Increase effective batch size if needed
        dataloader_num_workers=NUM_WORKERS,           # Adjust according to CPU cores
        disable_tqdm=False
    )

    trainer = TrainValMetricsTrainer(
        confmat_dir=CONFMAT_DIR,
        model=model,
        args=training_args,
        data_collator=collate_with_filename,
        train_dataset=train_subset,
        eval_dataset= val_subset,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(MetricsCallback(trainer=trainer, cf_dir=CONFMAT_DIR))
    trainer.add_callback(SaveBestPredictionsCallback(trainer=trainer, save_dir=BESTPREDS_DIR))
    trainer.add_callback(SavesCurrentStateCallback(trainer=trainer, last_checkpoint_dir=LAST_CHECKPOINT_DIR))

    # Train
    trainer.train(resume_from_checkpoint=EXISTING_DIR_TO_RESUME_FROM)

    # Save final model
    trainer.save_model(os.path.join(RESULTS_DIR, "segformer_trained_model"))
    processor.save_pretrained(os.path.join(RESULTS_DIR, "segformer_trained_model"))

    # Visualization
    last_checkpoint_path = trainer.state.best_model_checkpoint or trainer.state.last_model_checkpoint
    state_file = os.path.join(last_checkpoint_path, "trainer_state.json")
    with open(state_file, "r") as f:
        state = json.load(f)
    history = state["log_history"]

    show_loss_pa(history,os.path.join(IMG_DIR, 'loss_pa.png'), False, True)
    show_mean_iou_dice(history,os.path.join(IMG_DIR, 'mean_iou_dice.png'), False, True)
    show_iou_per_class(history,os.path.join(IMG_DIR, 'iou_per_class.png'), False, True)

    # Show duration of process
    delta_time_loop = time() - time_start
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"---\n\n==== Training completed in {hours}:{min}:{sec} ====\n")


if __name__ == "__main__":
    conf_train = OmegaConf.load('./config/train.yaml')
    conf_dataset = OmegaConf.load('./config/dataset.yaml')

    args= OmegaConf.merge({"train":conf_train, "dataset":conf_dataset})

    training_model(args)
