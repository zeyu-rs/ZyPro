# ZyPro - Deep Learning Framework for Remote Sensing Image Processing

## Overview
ZyPro is a deep learning framework specifically designed for remote sensing image processing and segmentation. It provides a comprehensive set of tools for training models, processing images, and performing semantic segmentation on remote sensing data.

## Project Structure
```
ZyPro_v1/
├── predict.py           # Prediction script for trained models
├── read_pth.py         # Utility for reading model weights
├── test.py             # Testing script
├── train.py            # Main training script
├── zycontrol/          # Training control modules
│   ├── lr.py          # Learning rate schedulers
│   ├── train_set.py   # Training set configuration
│   └── training.py    # Training loop implementation
├── zydata/            # Data processing modules
│   ├── pre_tif_large.py  # Large TIF file processing
│   ├── rsread.py      # Remote sensing data reader
│   └── rsseg.py       # Segmentation data processing
├── zyloss/            # Loss functions
│   ├── basic_loss.py  # Basic loss functions
│   └── loss_seg.py    # Segmentation-specific losses
├── zynets/            # Neural network architectures
│   └── unet.py        # UNet implementation
└── zytools/           # Utility tools
    ├── 3to1png.py     # Convert 3-channel to 1-channel PNG
    ├── arrange_s.py   # Dataset arrangement tools
    ├── clip_block.py  # Image clipping utilities
    ├── data_aug_jp.py # Data augmentation for JPG
    ├── hecheng_jp.py  # Image stitching tool
    ├── rename.py      # File renaming utility
    ├── resize_img_jp.py # Image resizing tool
    └── tif2jpg.py     # TIF to JPG converter
```

## Features
- **Image Processing**
  - Support for multiple image formats (TIF, JPG, PNG)
  - Image format conversion
  - Image resizing and clipping
  - Data augmentation
  - Image stitching
  
- **Deep Learning**
  - UNet architecture implementation
  - Custom loss functions
  - Flexible training pipeline
  - Learning rate scheduling
  - Model prediction tools

- **Remote Sensing**
  - Specialized remote sensing data processing
  - Semantic segmentation support
  - Large TIF file handling
  - Multi-channel image processing

## Requirements
- Python 3.x
- PyTorch
- GDAL
- NumPy
- PIL (Python Imaging Library)
- OpenCV (cv2)

## Usage

### Training
```python
# Configure training parameters in train.py
python train.py
```

### Testing
```python
# Configure test parameters in test.py
python test.py
```

### Prediction
```python
# Configure prediction parameters in predict.py
python predict.py
```

### Image Processing Tools
The `zytools` directory contains various utilities for image processing:
- Use `tif2jpg.py` for format conversion
- Use `data_aug_jp.py` for data augmentation
- Use `clip_block.py` for image clipping
- Use `hecheng_jp.py` for image stitching

## Model Architecture
The framework primarily uses the UNet architecture, which is particularly effective for semantic segmentation tasks in remote sensing applications. The implementation includes:
- Multiple UNet variants
- Customizable input/output channels
- Batch normalization
- Skip connections

## Loss Functions
Available loss functions include:
- BCE Loss
- Tversky Loss
- Combo Loss

## Data Processing
The framework supports:
- Multi-channel remote sensing data
- Large TIF file processing
- Batch processing
- Data augmentation (flipping, rotation)
- Image tiling and stitching

## Author
Zeyu Xu

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Notes
- All file paths should be configured in the respective scripts before running
- GPU support is available when CUDA is detected
- Ensure sufficient disk space for large dataset processing 