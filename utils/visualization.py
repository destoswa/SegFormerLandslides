import os
import json
import matplotlib.pyplot as plt

def show_loss_pa(history, saving_loc, do_show=False, do_save=True):
    # Lists to fill
    train_loss = []
    train_pa = []
    val_loss = []
    val_pa = []

    # Parse log history
    for entry in history:
        if "train_loss" in entry:
            train_loss.append(entry["train_loss"])

        if "train_pa" in entry:
            train_pa.append(entry["train_pa"])

        if "eval_pa" in entry:
            val_pa.append(entry["eval_pa"])

        if "eval_loss" in entry:
            val_loss.append(entry["eval_loss"])

    # -----------------------
    # Plot 1: Training Loss
    # -----------------------
    plt.figure(figsize=(14,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train')

    plt.plot(val_loss, label='val')
    plt.title("Loss")
    plt.legend()
    plt.xlabel("Epoch [-]")
    plt.ylabel("Loss [-]")

    # -----------------------
    # Plot 2: Validation Pixel accuracy
    # -----------------------
    plt.subplot(1, 2, 2)
    plt.plot(train_pa, label='train')
    plt.plot(val_pa, label='val')
    plt.title("Pixel Accuracy")
    plt.legend()
    plt.xlabel("Epoch [-]")
    plt.ylabel("PA [-]")
    
    plt.tight_layout()
    if do_save:
        plt.savefig(saving_loc)
        plt.savefig(saving_loc.split('.')[0] + '.eps', format='eps')
    
    if do_show:
        plt.show()


def show_mean_iou_dice(history, saving_loc, do_show=False, do_save=True):
    # Lists to fill
    train_mdice = []
    train_miou = []
    val_mdice = []
    val_miou = []

    # Parse log history
    for entry in history:
        if "train_loss" in entry:
            train_miou.append(entry["train_mean_iou"])

        if "train_pa" in entry:
            train_mdice.append(entry["train_mean_dice"])

        if "eval_pa" in entry:
            val_miou.append(entry["eval_mean_iou"])

        if "eval_loss" in entry:
            val_mdice.append(entry["eval_mean_dice"])

    # -----------------------
    # Plot 1: Training Loss
    # -----------------------
    plt.figure(figsize=(14,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_miou, label='train')

    plt.plot(val_miou, label='val')
    plt.title("Mean IoU")
    plt.legend()
    plt.xlabel("Epoch [-]")
    plt.ylabel("IoU [-]")

    # -----------------------
    # Plot 2: Validation Pixel accuracy
    # -----------------------
    plt.subplot(1, 2, 2)
    plt.plot(train_mdice, label='train')
    plt.plot(val_mdice, label='val')
    plt.title("Mean Dice")
    plt.legend()
    plt.xlabel("Epoch [-]")
    plt.ylabel("Dice [-]")
    
    plt.tight_layout()

    if do_save:
        plt.savefig(saving_loc)
        plt.savefig(saving_loc.split('.')[0] + '.eps', format='eps')
    
    if do_show:
        plt.show()
    

def show_iou_per_class(history, saving_loc, do_show=False, do_save=True):
    # Per-class IoU (dynamic)
    val_per_class_iou = {}  # {class_id: [iou_epoch1, iou_epoch2, ...]}
    train_per_class_iou = {}  # {class_id: [iou_epoch1, iou_epoch2, ...]}

    # Parse log history
    for entry in history:
        # Per-class IoU
        for key, value in entry.items():
            if key.startswith("eval_iou_class_"):
                cls = int(key.split("_")[-1])
                val_per_class_iou.setdefault(cls, [])
                val_per_class_iou[cls].append(value)
            elif key.startswith("train_iou_class_"):
                cls = int(key.split("_")[-1])
                train_per_class_iou.setdefault(cls, [])
                train_per_class_iou[cls].append(value)
    
    # Show results
    plt.figure(figsize=(14,5))
    plt.subplot(1, 2, 1)
    mapping_class_names = {key: val for key,val in zip(train_per_class_iou.keys(), ['Background', 'Landslide'])}
    for cls, values in train_per_class_iou.items():
        plt.plot(values, label=mapping_class_names[cls])
    plt.title("Training set")
    plt.xlabel("Epoch [-]")
    plt.ylabel("IoU [-]")
    plt.legend()

    plt.subplot(1, 2, 2)
    mapping_class_names = {key: val for key,val in zip(val_per_class_iou.keys(), ['Background', 'Landslide'])}
    for cls, values in val_per_class_iou.items():
        plt.plot(values, label=mapping_class_names[cls])
    plt.title("Validation set")

    plt.xlabel("Epoch [-]")
    plt.ylabel("IoU [-]")
    plt.legend()

    plt.suptitle("Mean IoU per class")
    plt.tight_layout()
    
    if do_save:
        plt.savefig(saving_loc)
        plt.savefig(saving_loc.split('.')[0] + '.eps', format='eps')
    
    if do_show:
        plt.show()


if __name__ == "__main__":
    src = r"outputs\20251208_162944_10_epochs_b0_dataset_CAS_2500\checkpoint-330"
    # visualize_training(src)
    # Path to trainer_state.json
    state_file = os.path.join(src, "trainer_state.json")
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"trainer_state.json not found in {src}")

    # Load training state
    with open(state_file, "r") as f:
        state = json.load(f)

    history = state["log_history"]
    # print(history[-1])
    show_mean_iou_dice(history)
    # show_iou_per_class(history)
