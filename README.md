# FloodVision — Flood Detection Using Semantic Segmentation with U-Net

A student-friendly end-to-end computer-vision project. It includes a Flask website, a PyTorch U-Net model, a training script, image upload, segmentation-mask generation, overlay visualization, and predicted flood-pixel coverage.

> **Important:** The ZIP does not contain trained model weights or a flood dataset. The website starts immediately, but real AI predictions become available only after you add labelled training data and run training. Never present random/untrained output as a real prediction.

## 1. Features

- Upload an aerial/satellite RGB image (PNG, JPG, TIFF)
- U-Net pixel-level binary segmentation: flood vs background
- Show original image, predicted mask, and overlay
- Estimate the percentage of pixels classified as flood
- Train/validation data folder structure
- Automatically saves the best checkpoint

## 2. Requirements

- Python 3.10 or newer
- 8 GB RAM recommended
- CPU works for small experiments; NVIDIA GPU is recommended for training
- Internet connection for installing Python packages

## 3. Installation (Windows)

Open Command Prompt in this project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation reports a hardware/CUDA issue, install the CPU build from the official PyTorch installation selector: https://pytorch.org/get-started/locally/

## 4. Prepare dataset

Place paired images and masks in the folders shown below:

```text
data/
  train/
    images/
      scene_001.jpg
      scene_002.jpg
    masks/
      scene_001.png
      scene_002.png
  val/
    images/
    masks/
```

Masks must be grayscale: white pixels represent flood, black pixels represent background. Each image must have a matching mask with the same filename stem. Use expert-labelled masks and make sure the training, validation, and test sets are separated by location/event where possible to reduce data leakage.

For an academic RGB aerial-image project, investigate the FloodNet dataset and its usage terms. For satellite SAR research, Sen1Floods11 is a common benchmark, but this starter code expects 3-channel RGB images; SAR VV/VH data requires adapting the input channels and raster preprocessing.

## 5. Train the U-Net model

```bash
python train.py
```

The script trains for 20 epochs and saves the best validation checkpoint to:

```text
models/best_unet.pth
```

If no validation images are present, it uses training loss to select a checkpoint. For a final report, use a separate test set and report IoU, Dice/F1, precision, and recall; do not claim performance values without running evaluation.

## 6. Run the website

```bash
python -m app.main
```

Open this address in your browser:

http://127.0.0.1:5000

Upload an image after the trained checkpoint exists.

## 7. Project workflow

1. Collect labelled flood imagery.
2. Resize images and masks to 256 × 256.
3. Train U-Net using binary cross-entropy + Dice loss.
4. Save the best checkpoint.
5. Upload an unseen image through the Flask website.
6. Apply sigmoid and a 0.5 threshold to generate a binary mask.
7. Display the mask and overlay; calculate predicted flood coverage.

## 8. Main project modules

- `app/model.py` — U-Net architecture
- `train.py` — dataset loader, training loop, loss, checkpoint saving
- `app/main.py` — Flask routes and inference pipeline
- `app/templates/index.html` — website layout
- `app/static/css/style.css` — responsive styling
- `app/static/js/app.js` — upload and prediction UI

## 9. Limitations

This is an academic prototype, not a certified flood-warning system. Prediction quality depends on the dataset, label quality, sensor type, geography, season, and training. Permanent water, shadows, cloud, vegetation, and urban surfaces can cause errors. Flood coverage is image-pixel coverage, not geographic area. Do not use it for emergency decisions without expert and authoritative validation.

## 10. Dataset and research references

- Sen1Floods11 paper: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf
- Sen1Floods11 project: https://github.com/cloudtostreet/Sen1Floods11
- U-Net paper: https://arxiv.org/abs/1505.04597
- PyTorch: https://pytorch.org/

## 11. Suggested report sections

Abstract · Introduction · Problem Statement · Objectives · Existing System · Proposed System · System Architecture · Dataset Description · U-Net Architecture · Implementation · Results and Discussion · Limitations · Conclusion · References


## Publish online
See `DEPLOY_ON_RENDER.md` for the Render + GitHub deployment steps. The included `render.yaml` and `Procfile` configure the web service. You must create and authorize your own hosting account to receive a public URL.
