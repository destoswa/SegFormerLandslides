import os
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
from utils.trainer import TrainValMetricsTrainer
from utils.metrics import compute_metrics
from utils.callbacks import PrinterCallback


def training_model(args):
    OUTPUT_SUFFIXE = args.train.output_suffixe
    VAL_SPLIT = args.train.val_split
    NUM_EPOCHS = args.train.num_epochs
    NUM_WORKERS = args.train.num_workers
    BATCH_SIZE = args.train.batch_size
    LEARNING_RATE = args.train.learning_rate
    WEIGHT_DECAY = args.train.weight_decay
    PRETRAINED_MODEL = args.train.pretrained_model
    DATASET_DIR = args.dataset.dataset_dir

    OUTPUT_DIR = datetime.now().strftime(r"%Y%m%d_%H%M%S_") + OUTPUT_SUFFIXE
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    time_start = time()

    # ----------------------------------------------------------
    # 2) Load model + processor
    # ----------------------------------------------------------

    processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL, use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL,
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
        image_dir=os.path.join(DATASET_DIR, "images"),
        mask_dir=os.path.join(DATASET_DIR, "masks"),
        processor=processor
    )

    # print("training set size: ", len(train_ds))

    # Split train/val
    split_idx = int(len(train_ds) * (1 - VAL_SPLIT))
    train_subset, val_subset = torch.utils.data.random_split(train_ds, [split_idx, len(train_ds) - split_idx])


    # ----------------------------------------------------------
    # 4) Training arguments
    # ----------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,  # Where checkpoints and logs are saved
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
        # report_to=["tensorboard"],        # Can add "wandb" if desired

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

    # trainer = Trainer(
    trainer = TrainValMetricsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset= val_subset,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(PrinterCallback(trainer))

    # ----------------------------------------------------------
    # 6) Train
    # ----------------------------------------------------------
    trainer.train()

    # ----------------------------------------------------------
    # 7) Save final model
    # ----------------------------------------------------------
    trainer.save_model(os.path.join(OUTPUT_DIR, "segformer_trained_model"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "segformer_trained_model"))

    # ----------------------------------------------------------
    # 8) Visualization
    # ----------------------------------------------------------
    # visualization(trainer)


    # Show duration of process
    delta_time_loop = time() - time_start
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"---\n\n==== Training completed in {hours}:{min}:{sec} ====\n")


if __name__ == "__main__":
    print(torch.cuda.is_available())
    conf_train = OmegaConf.load('./config/train.yaml')
    conf_dataset = OmegaConf.load('./config/dataset.yaml')

    args= OmegaConf.merge({"train":conf_train, "dataset":conf_dataset})

    # training_model(args)