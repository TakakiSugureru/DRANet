import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from DRANet import DRANet
import argparse

# ==== Preprocessing function ====
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image).unsqueeze(0)  # Shape: (1, C, H, W)
    return tensor, image.size

def save_image(tensor, save_path, size=None):
    img = tensor.squeeze(0).clamp(0, 1).cpu()
    img_pil = TF.to_pil_image(img)
    if size:
        img_pil = img_pil.resize(size, Image.BICUBIC)
    img_pil.save(save_path)

# ==== PSNR calculation ====
def calc_psnr(output, target, max_val=1.0):
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# ==== MAIN TEST FUNCTION ====
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DRANet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate over all images in the input directory
    image_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for path in image_paths:
        filename = os.path.basename(path)
        noisy_tensor, size = load_image(path)
        noisy_tensor = noisy_tensor.to(device)

        with torch.no_grad():
            output_tensor = model(noisy_tensor)

        save_path = os.path.join(args.output_dir, filename)
        save_image(output_tensor, save_path, size=size)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    class Args:
        model_path = '/content/drive/MyDrive/DRANet/Model/model_dranet_guass.pth'  # Path to the trained model
        input_dir = '/content/drive/MyDrive/DRANet/Data/Test/Clean'                # Path to the input images
        output_dir = '/content/drive/MyDrive/DRANet/test_output'                   # Path to save denoised images

    args = Args()
    main(args)
