# Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration

This repository provides the official implementation of the paper:

**Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration**  
Accepted to **CVPR 2026**.

---

## Overview

This project focuses on test-time adaptation for all-in-one image restoration.  
Given a pre-trained all-in-one restoration model, our method performs degradation-consistent adaptation on test data and then automatically evaluates the adapted model.

---

## Environment Setup

Please install the required dependencies according to `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Pre-trained Models

The pre-trained weights are provided in the `pretrain/` folder.

```bash
pretrain/
├── model.ckpt
└── epoch=80.ckpt
```

- `model.ckpt`: pre-trained PromptIR model on the 3-task setting.
- `epoch=80.ckpt`: pre-trained PromptIR model on the 5-task setting.

Please make sure the corresponding checkpoint files are placed under the `pretrain/` directory before running testing or adaptation.

---

## Testing the Original PromptIR Model

We provide the testing script for the original PromptIR model:

```bash
python test_promptir.py
```

This script evaluates the original PromptIR model directly on the test datasets **without test-time adaptation**.

---

## Test-Time Adaptation

The adaptation process is performed by:

```bash
python train_test_promptir.py
```

After the test-time adaptation process is completed, the adapted model will be automatically evaluated on the corresponding test dataset.

---


## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{tang2026degradation,
  title={Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration},
  author={{Tang, Ni and Nie, Shenghao and Luo, Xiaotong and Xie, yuan and Qu, Yanyun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15476--15485},
  year={2026}
}
```

---

## Acknowledgement

This project is built upon [[PromptIR](https://github.com/va1shn9v/PromptIR)]. We sincerely thank the authors for their excellent work.
