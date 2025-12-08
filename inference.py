import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from torchvision.transforms.functional import to_pil_image


def load_latest_checkpoint(model_dir):
    """
    Returns the path to the latest checkpoint folder inside model_dir.
    If none found, return model_dir (trained_model directory).
    """
    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    ckpts = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        print("[INFO] No checkpoints found. Using main model directory.")
        return model_dir

    # Sort by step number
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    last_ckpt = ckpts_sorted[-1]

    print(f"[INFO] Using checkpoint: {last_ckpt}")
    return os.path.join(model_dir, last_ckpt)


def predict_image(model, processor, image_path, device="cuda"):
    """
    Runs inference on a single image and returns:
    - predicted_mask (H, W) with class indices
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Preprocess
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, num_classes, h, w)

    # Resize logits to original size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )

    pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    return pred_mask


def run_inference(
        model_dir,
        data_dir,
        device="cuda"
):
    """
    Runs inference on all images in a directory.
    """

    # ----------------------------
    # Load best checkpoint
    # ----------------------------
    ckpt_path = load_latest_checkpoint(model_dir)

    processor = AutoImageProcessor.from_pretrained(ckpt_path)
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt_path)
    model.to(device)
    model.eval()

    output_dir = os.path.join(data_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # Loop over images
    # ----------------------------
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    image_list = [f for f in os.listdir(data_dir) if f.lower().endswith(exts)]

    if not image_list:
        print("No images found in:", data_dir)
        return

    print(f"[INFO] Running inference on {len(image_list)} images")

    for _, img_name in tqdm(enumerate(image_list), total=len(image_list), desc="Predicting"):
        input_path = os.path.join(data_dir, img_name)
        output_path = os.path.join(output_dir, img_name.replace(".jpg", ".png"))

        mask = predict_image(model, processor, input_path, device=device)
        pil_mask = Image.fromarray(mask.astype(np.uint8), mode="L")
        pil_mask.save(output_path)

        # print(f"Saved mask: {output_path}")


if __name__ == "__main__":
    args = OmegaConf.load('config/inference.yaml')

    run_inference(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
    )
