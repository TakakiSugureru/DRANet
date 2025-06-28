# DRANet - Denoising Residual Attention Network

This repository implements the **DRANet** model for image denoising using deep convolutional neural networks with attention mechanisms. The project is built and tested on **Google Colab**.

---

## Original Reference

- Paper: [DRANet: Deep Residual Attention Network for Image Denoising](https://arxiv.org/abs/2303.06750)
- Original author’s code: [https://github.com/WenCongWu/DRANet](https://github.com/WenCongWu/DRANet)

---

## Project Overview

- **`makefile.ipynb`**:  
  Notebook for generating key files:
  - `dataset.py`: defines the dataset for training/validation
  - `DRANet.py`: defines the DRANet architecture
  - `test.py`: script for testing denoising results

- **`Run.ipynb`**:  
  Contains core code:
  - Model training
  - Inference/testing
  - Image loading and PSNR/SSIM visualization

---

## Datasets Used

- **Training set**: [CBSD68 Dataset](https://github.com/clausmichele/CBSD68-dataset)  
  (You need to download it, and load the CBSD68 folder into the Data -> Train)

- **Test set**: Random images from the internet or user-provided samples.

---

## How to Use (in `Run.ipynb`)

### **Step 1: Load Dataset**
- Set the variable `root_dir`.
- Optionally: train the DRANet model or load a pre-trained one.


### **Step 2: Add Gaussian Noise to Clean Images**
- Apply Gaussian noise with level `sigma=25` to the clean images.
- You must match the training noise level. For example, if the model was trained on `sigma=25`, test images should use the same or lower noise level.
- You can skip this step if noisy images with the correct noise level are already available.
- Important: DRANet was trained on a fixed noise level `(sigma=25)`. Using a higher level during testing `(e.g. sigma=50)` may lead to poor results.

### **Step 3: Run Testing**
- **Option 1**: Denoise all images in a folder (via `makefile.ipynb`)
  - Adjust paths in `test.py`
- **Option 2**: Run test directly in `Run.ipynb`
  - PSNR, SSIM and visual comparisons can be displayed

---

## Environment

- **Google Colab**
- Dependencies:
  - `PyTorch`
  - `Torchvision`
  - `scikit-image`
  - `matplotlib`, `PIL`
