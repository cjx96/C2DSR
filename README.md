C2DSR
===

The source code is for the paper: “Contrastive Cross-Domain Sequential Recommendation” accepted in CIKM 2022 by Jiangxia Cao,  Xin Cong, Jiawei Sheng, Tingwen Liu and Bin Wang.

```
@inproceedings{cao2022c2dsr,
  title={Contrastive Cross-Domain Sequential Recommendation},
  author={Cao, Jiangxia and Cong, Xin and Sheng, Jiawei and Liu, Tingwen and Wang, Bin},
  booktitle={ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2022}
}
```

Requirements
---

Python=3.7.9

PyTorch=1.6.0

Scipy = 1.5.2

Numpy = 1.19.1

### Dasetsets

We find the information leak issue from existed datasets which released by previous works  (pi-net and MIFN), and our corrected versions are provided [here](https://drive.google.com/drive/folders/1xpnp6tH56xz8PF_xuTi9exEptmcvlAVU?usp=sharing).

Note that all datasets are required in the `./dataset` directory.

Usage
---

To run this project, please make sure that you have the following packages being downloaded. Our experiments are conducted on a PC with an Intel Xeon E5 2.1GHz CPU, 256 RAM and a Tesla T4 16GB GPU. 

Running example:

```shell
CUDA_VISIBLE_DEVICES=0 python -u train_rec.py --undebug  --data_dir Food-Kitchen > Food-Kitchen.log 2>&1&
```
