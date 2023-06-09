# SemLA
Thank you for your attention😀! The code is being organized, and the complete project will be released before July 1.

## Data preparation
1. Download the [COCO](https://drive.google.com/drive/folders/1rN5o903LXiIq54IvgxGLJnfb_f1jtoMt?usp=share_link) dataset to ```.\datasets\COCO\``` (path2COCO)
2. Download the [IVS](https://github.com/xiehousheng/IVS_data) dataset to ```.\datasets\IVS\``` (path2IVS)
3. Download the label of [IVS](https://github.com/xiehousheng/IVS_data) dataset to ```.\datasets\IVS_Label\``` (path2IVS_Label)
4. Generate pseudo-infrared images for each image in the COCO dataset using [CPSTN](https://github.com/wdhudiekou/UMF-CMGR/tree/main/CPSTN) and store the results in ```.\datasets\COCO_CPSTN\``` (path2COCO_CPSTN)
5. Generate pseudo-infrared images for each image in the IVS dataset using [CPSTN](https://github.com/wdhudiekou/UMF-CMGR/tree/main/CPSTN) and store the results in ```.\datasets\IVS_CPSTN\``` ((path2IVS_CPSTN))

## Training
1. Train stage1: Registration and semantic feature extraction. ```cd train_stage1``` and configuring dataset paths, then run ```python train_stage1.py```

2. Train stage2: Training CSC and SSR modules. ```cd train_stage2``` and configuring dataset paths, then run ```python train_stage2.py```

3. Train stage3: Training fusion module. ```cd train_stage3``` and configuring dataset paths, then run ```python train_stage3.py```

## Test
Download pre-trained models on [Google Drive](https://drive.google.com/drive/folders/1Lh9UFXWP5bvt_MVwYa9ZPA7g_lOEGrxz?usp=drive_link) or [Baidu Yun](https://pan.baidu.com/s/1CBG0k0PJMqHQsHybdMkRow?pwd=5fmn) and configure the path ```reg_weight_path```, ```fusion_weight_path```. We provide two matching modes, one is semantic object-oriented matching, setting '''matchmode = "semantic"''', and the other is global image oriented matching, setting '''matchmode = "scene"'''.
### On a dataset
Configuring dataset paths, then run ```python test.py```
### On a pair of images
Configuring images path, then run ```python inference_one_pair_images.py```

## Citation

If this code is useful for your research, please cite our paper.
```bibtex
@article{xie2023semantics,
  title={Semantics lead all: Towards unified image registration and fusion from a semantic perspective},
  author={Xie, Housheng and Zhang, Yukuan and Qiu, Junhui and Zhai, Xiangshuai and Liu, Xuedong and Yang, Yang and Zhao, Shan and Luo, Yongfang and Zhong, Jianbo},
  journal={Information Fusion},
  pages={101835},
  year={2023},
  publisher={Elsevier}
}
```
