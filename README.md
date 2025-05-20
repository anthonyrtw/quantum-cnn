# Hamming weight preserving quantum convolutional neural networks

In this notebook I attempt to reproduce results from Monbroussou et al titled [Subspace preserving quantum convolutional neural network architectures](https://iopscience.iop.org/article/10.1088/2058-9565/adbf43#back-to-top-target). Here the authors propose a novel quantum convolutional neural network (QCNN) where each layer preserves the hamming weight distance of the model. i.e. the number of $|1\rangle$ states in the model statevector. 

This allows for the QCNN to be efficiently trainable, escaping the problem of barren plateaus which is endemic to many quantum machine learning models.


## Structure 

```bash
├── data/ # Download the MNIST dataset here.
├── figures/ 
├── logs/
│   ├── 8x8 C(k=2, s=2)-P(k=2)-D-L T=0.015 batch_size=16 lr=0.001/
│   │   └── history.npz # Logged training, validation and loss curves.
│   │   └── qcnn_weights.pth # Optimized QCNN weights.
    ...
├── training_script.ipynb
├── training_script.py
├── src.py
├── requirements.txt
└── README.md
```

The logging of this project is notated by the structure of the QCNN and the hyperparameters used. For instance, consider the following directory:
```logs\8x8 C(k=2, s=2)-P(k=2)-PD-L T=0.015 batch_size=16 lr=0.001```

QCNN Structure:
- ```C(k=2, s=2)```: Convolution kernel size $= 2$, stride $= 2$. <br>
- ```P(k=2)```: Pooling kernel size $= 2$, stride $= 2$ <br>
- ```PD```: Pyramidal dense layer. <br>
- ```L```: Classical linear layer 

Hyperparameters:
- ```T=0.015```: 0.015 Softmax temperature. <br>
- ```batch_size=16```: 16 batch size for training and evaluation. <br>
- ```lr=0.001```: Learning rate $= 0.001$. <br>


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
