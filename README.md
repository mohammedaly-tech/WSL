
# Thyroid Segmentation with TG3K and TN3K Datasets

This repository contains the implementation of a thyroid nodule segmentation model using **U-Net** and the **AdamW** optimizer. The model is trained on the **TG3K** dataset and tested on the **TN3K** dataset.

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Thyroid-Segmentation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the **TG3K** and **TN3K** datasets:
   - TG3K: [Link](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TG3K.md)
   - TN3K: [Link](https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view)

4. Prepare your data:
   - Place the TG3K images and masks in the respective directories under `data/tg3k/`.
   - Place the TN3K images and masks in the respective directories under `data/tn3k/`.

## Training

To train the model:
```bash
python train.py
```

## Testing

After training, to test the model on the **TN3K** dataset:
```bash
python train.py --test
```

## Model Architecture

We use a U-Net architecture for segmentation tasks. The model has been implemented in PyTorch.

## Citing

If you find this work useful, please consider citing our paper and code:

```
@article{your_paper_reference,
  title={Thyroid Region Segmentation with U-Net on TG3K and TN3K},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```
