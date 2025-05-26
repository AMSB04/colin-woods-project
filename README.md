# Colin Woods Project
 The codes for the work "A Novel CycleGAN Framework for 7T MRI Synthesis: Toward Enhanced Epilepsy Visualisation"
 
 ## Overview
This repository contains a custom Swin Transformer V2-based U-Net generator designed for high-fidelity synthesis of 7T MRI from 3T MRI axial slices. The generator is integrated into a CycleGAN framework, aiming to improve the visualisation of epileptogenic features in lower-field MRI.

## Directory Structure
```
colin-woods-project/
├── configs/
│   └── config.yaml
├── data/
│   └── .gitkeep
├── networks/
│   ├── discriminator.py
│   ├── generator.py
├── utils/
│   ├── logger.py
│   ├── model_util.py
├── .gitattributes
├── .gitignore
├── dataset.py
├── evaluate.py
├── LICENSE
├── README.md
├── requirements.txt
└── train.py
```

## Data
Organise dataset in the following structure:
```
dataset_root/
├── 3T/
│   ├── train/
│   ├── val/
│   └── test/
└── 7T/
    ├── train/
    ├── val/
    └── test/
```

## Preprocessing Assumptions
Before feeding into the model:
- Bias field corrected (FSL FAST)
- Spatially normalized to MNI152 (SPM)
- Intensity normalized to [-1, 1] (SPM image calculator)
- Skull stripped (FSL BET)
- 2D axial slices extracted (FSL `fslslice`)
- Aligned paired slices for supervised loss

## Getting Started
Training and evaluation scripts are in progress. 7T/3T Datasets are required for further testing. 

## Notes
This project was developed as part of a submission for the **Colin Woods Prize**. The long-term goal is to explore how synthetic 7T images can enhance epilepsy diagnostics using accessible 3T data. Please refer to "A Novel CycleGAN Framework for 7T MRI Synthesis: Toward Enhanced Epilepsy Visualisation" for more information.

## References
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet)
- [CycleGAN](https://arxiv.org/abs/1703.10593)