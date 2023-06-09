# SemLA
Thank you for your attention😀! The code is being organized, and the complete project will be released before July 1.

## Data preparation
1. Download the [COCO](https://drive.google.com/drive/folders/1rN5o903LXiIq54IvgxGLJnfb_f1jtoMt?usp=share_link) dataset to ```.\datasets\COCO\``` (path2COCO)
2. Download the [IVS](https://github.com/xiehousheng/IVS_data) dataset to ```.\datasets\IVS\``` (path2IVS)
3. Download the label of [IVS](https://github.com/xiehousheng/IVS_data) dataset to ```.\datasets\IVS_Label\``` (path2IVS_Label)
4. Generate pseudo-infrared images for each image in the COCO dataset using [CPSTN](https://github.com/wdhudiekou/UMF-CMGR/tree/main/CPSTN) and store the results in ```.\datasets\COCO_CPSTN\``` (path2COCO_CPSTN)
5. Generate pseudo-infrared images for each image in the IVS dataset using [CPSTN](https://github.com/wdhudiekou/UMF-CMGR/tree/main/CPSTN) and store the results in ```.\datasets\IVS_CPSTN\``` ((path2IVS_CPSTN))

## Training
1. Train stage1: Registration and semantic feature extraction.  
```cd train_stage1``` and Configuring dataset paths, then ```python train_stage1.py``
2. Train stage2: Training CSC and SSR modules.
```cd train_stage2``` and Configuring dataset paths, then ```python train_stage2.py``
3. Train stage1: Training fusion module.
```cd train_stage3``` and Configuring dataset paths, then ```python train_stage3.py``
