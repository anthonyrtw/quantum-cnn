# Hamming weight preserving quantum convolutional neural network

In this notebook I attempt to reproduce results from Monbroussou et al titled [Subspace preserving quantum convolutional neural network architectures](https://iopscience.iop.org/article/10.1088/2058-9565/adbf43#back-to-top-target). Here the authors propose a novel quantum convolutional neural network (QCNN) where each layer preserves the hamming weight distance of the model. i.e. the number of $|1\rangle$ states in the model statevector. 

This allows for the QCNN to be efficiently trainable, escaping the problem of barren plateaus which is endemic to many quantum machine learning models.

## Structure 

```bash
├── data/ # Download the MNIST dataset here.
├── figures/ 
├── logs/
│   ├── 8x8 C(k=2, s=2)-P(k=2)-D-L T=0.015 batch_size=16 lr=0.001/ # Each log describes the model structure and hyperparameters used.
│   │   └── history.npz # Logged training, validation and loss curves.
│   │   └── qcnn_weights.pth # Optimized QCNN weights.
    ...
├── training_script.ipynb
├── training_script.py
├── src.py
├── requirements.txt
└── README.md
```

## Requirements

This repository mostly relies on the [Torch Quantum](https://github.com/mit-han-lab/torchquantum) python package. Additionally the following python packages were used:

```bash
matplotlib==3.10.3
numpy==2.2.5
torch==2.5.1
torchquantum.egg==info
torchvision==0.20.1
```

Alternatively run the following command:
```bash
pip install requirements.txt
```
