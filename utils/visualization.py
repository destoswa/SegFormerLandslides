import os
import json
import matplotlib.pyplot as plt


def visualize_training(run_dir):
    """
    Loads HuggingFace Trainer logs from a checkpoint folder and plots:
        - Training Loss
        - Validation Loss
        - Validation Mean IoU
        - (Optional) IoU per class
    
    Args:
        run_dir (str): Path to the training output directory containing trainer_state.json.
                       Example: "./segformer_output_2"
    """

    # Path to trainer_state.json
    state_file = os.path.join(run_dir, "trainer_state.json")
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"trainer_state.json not found in {run_dir}")

    # Load training state
    with open(state_file, "r") as f:
        state = json.load(f)

    history = state["log_history"]

    # Lists to fill
    train_loss = []
    val_loss = []
    val_miou = []
    val_miou_ep = []  # epoch index for plotting
    val_loss_ep = []

    # Per-class IoU (dynamic)
    per_class_iou = {}  # {class_id: [iou_epoch1, iou_epoch2, ...]}

    step_counter = 0
    epoch_counter = 0

    print(history)

    # Parse log history
    for entry in history:
        # Training loss (logged every logging_steps)
        if "loss" in entry:
            train_loss.append(entry["loss"])
            step_counter += 1

        # Validation metrics (logged every epoch)
        if "eval_mean_iou" in entry:
            val_miou.append(entry["eval_mean_iou"])
            val_miou_ep.append(epoch_counter)

        if "eval_loss" in entry:
            val_loss.append(entry["eval_loss"])
            val_loss_ep.append(epoch_counter)
            epoch_counter += 1

        # Per-class IoU
        for key, value in entry.items():
            if key.startswith("eval_iou_class_"):
                cls = int(key.split("_")[-1])
                per_class_iou.setdefault(cls, [])
                per_class_iou[cls].append(value)

    # -----------------------
    # Plot 1: Training Loss
    # -----------------------
    plt.figure(figsize=(14,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss)

    x = range(0,len(train_loss),len(train_loss)//(len(val_loss)-1))

    plt.plot(x, val_loss)
    plt.title("Training Loss")
    plt.xlabel("Logging Steps")
    plt.ylabel("Loss")

    # -----------------------
    # Plot 2: Validation Mean IoU
    # -----------------------
    plt.subplot(1, 2, 2)
    plt.plot(val_miou_ep, val_miou)
    # plt.plot(train_Mio, val_miou)
    plt.title("Validation Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")

    # # -----------------------
    # # Plot 3: Validation Loss
    # # -----------------------
    # plt.subplot(1, 3, 3)
    # plt.plot(val_loss_ep, val_loss)
    # plt.title("Validation Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------
    # Optional: Plot IoU per class in new figure
    # ------------------------------------------
    if per_class_iou:
        plt.figure(figsize=(8, 6))

        mapping_class_names = {key: val for key,val in zip(per_class_iou.keys(), ['Background', 'Landslide'])}
        for cls, values in per_class_iou.items():
            plt.plot(values, label=mapping_class_names[cls])

        plt.title("IoU Per Class (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()
        plt.show()

    print("Visualization complete!")


if __name__ == "__main__":
    src = r"D:\GitHubProjects\Terranum_repo\LandSlides\CustomSegFormer\segformer_output_2\checkpoint-11265"
    visualize_training(src)
