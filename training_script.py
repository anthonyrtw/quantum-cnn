# In[]
# ==== IMPORTS & SET-UP ====

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from torchquantum import QuantumDevice
from src import OneHotAmplitude, QConv2d, QPooling, OneHotAmplitude, PyramidQDense, MeasureLayer

import os
import numpy as np
from scipy.ndimage import uniform_filter 
import matplotlib.pylab as plt

plt.rcParams['legend.fontsize'] = 15
plt.rcParams['xtick.labelsize'] = 15 
plt.rcParams['ytick.labelsize'] = 15 
plt.rcParams['axes.labelsize'] = 20 
plt.rcParams['axes.titlesize'] = 19 
plt.rc('font', **{'family': 'serif', 'serif':['Computer Modern']})
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 500 


# In[]
# ==== HYPERPARAMETERS ====
dims = (8, 8)
batch_size = 16
temperature = 1
lr = 0.001
num_epochs = 40

torch.manual_seed(42)
CHECKPOINT = False


# In[]
# ==== Data loading ====

# Define transforms
transform = transforms.Compose([
    transforms.Resize(dims),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
])

# Load training and test datasets
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
val_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

# Consider 1/10th of the datasets.
train_indices = torch.randperm(60_000)[:6000]
val_indices = torch.randperm(10_000)[:1000]

train_dataset = Subset(train_dataset, train_indices.tolist())
val_dataset = Subset(val_dataset, val_indices.tolist())

# Dataloader for training QCNN
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True
)


# In[]
# ==== DEFINE MODEL ====

class QCNN(torch.nn.Module):
    """ 
    Quantum Convolutional Neural Network 

    QCNN consisting of quantum encoding, convolutional, pooling layers and 
    a classical linear layer to project the data onto the output dimension.

    :param dims: Input image dimensions.
    :param temperature: Softmax temperature used to evaluate the cross entropy loss.
    """

    def __init__(self, dims, temperature: float = 1.0):
        super().__init__()
        self.dims = dims
        self.temperature = temperature

        self._encoder = OneHotAmplitude(dims)
        self._qconv1 = QConv2d(dims, kernel_size=2)
        self._qpool1 = QPooling(dims, kernel_size=2)
        self._qdense = PyramidQDense(n_wires=sum(dims) // 2)
        self._measure = MeasureLayer(n_wires=sum(dims) // 2)

        # Convert to MNIST dimension.
        self.num_measurements = len(list(self._measure.output_states()))
        self._postprocess = torch.nn.Linear(self.num_measurements, 10, bias=True)

    def forward(self, X):
        if X.shape[-2:] != self.dims:
            raise ValueError('Image dimensions do not match QCNN dimensions.')

        # Remove channels dimension
        if len(X.shape) == 4:
            X = X.squeeze(1)

        # Pass through network
        qdev = QuantumDevice(n_wires=sum(X.shape[1:]))
        self._encoder(qdev, X) 
        self._qconv1(qdev) 
        self._qpool1(qdev)    
        self._qdense(qdev)
        
        # Measure & perform post-processing
        probs = self._measure(qdev)
        output = probs / self.temperature
        output = self._postprocess(output)
        return output

temperature = 0.015
qcnn = QCNN(dims, temperature)

num_params = sum(p.numel() for p in qcnn.parameters() if p.requires_grad)
print(f'QCNN has {num_params} parameters, \n')


# In[] 
# ==== LOAD CHECKPOINT ==== 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if CHECKPOINT:
    # Load weights and histories from logs.
    model_description = f'C(k=2, s=2)-P(k=2)-PD-L'
    log_dir = f'logs/{dims[0]}x{dims[1]} {model_description} T={temperature} batch_size={batch_size} lr={lr}' 

    with np.load(log_dir + '/history.npz') as data:
        train_losses_qcnn = list(data['train_losses'])
        train_accuracies_qcnn = list(data['train_accuracies'])
        val_accuracies_qcnn = list(data['val_accuracies'])

    qcnn.load_state_dict(torch.load(log_dir + '/qcnn_weights.pth', weights_only=False))

# Perform evaluation before training
else:
    train_losses_qcnn = []
    train_accuracies_qcnn = []
    val_accuracies_qcnn = [] 

    qcnn.eval()
    with torch.no_grad():
        running_correct = 0
        train_total_samples = 0
        val_running_correct = 0
        val_total_samples = 0

        # Evaluate first total accuracy over the entire training and validation sets
        for train_images, train_labels in train_loader: 
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            train_outputs = qcnn(train_images)

            _, train_preds = torch.max(train_outputs, 1)
            running_correct += (train_preds == train_labels).sum().item()
            train_total_samples += train_labels.size(0)

        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = qcnn(val_images)

            _, val_preds = torch.max(val_outputs, 1)
            val_running_correct += (val_preds == val_labels).sum().item()
            val_total_samples += val_labels.size(0)

    train_accuracy = running_correct / train_total_samples
    val_accuracy = val_running_correct / val_total_samples

    train_accuracies_qcnn = [train_accuracy]
    val_accuracies_qcnn = [val_accuracy]


# In[]
# ==== TRAINING ====
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(qcnn.parameters(), lr=lr)

if CHECKPOINT: # Check if loading from a checkpoint
    train_losses_qcnn = []
    train_accuracies_qcnn = []
    val_accuracies_qcnn = [] 

    # Evaluate first total accuracy over the entire training and validation sets
    qcnn.eval()
    with torch.no_grad():
        running_correct = 0
        total_samples = 0
        val_running_correct = 0
        val_total_samples = 0

        for train_images, train_labels in train_loader: 
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            train_outputs = qcnn(train_images)

            _, train_preds = torch.max(train_outputs, 1)
            running_correct += (train_preds == train_labels).sum().item()
            total_samples += train_labels.size(0)

        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = qcnn(val_images)

            _, val_preds = torch.max(val_outputs, 1)
            val_running_correct += (val_preds == val_labels).sum().item()
            val_total_samples += val_labels.size(0)

    train_accuracy = running_correct / total_samples
    val_accuracy = val_running_correct / val_total_samples

    train_accuracies_qcnn = [train_accuracy]
    val_accuracies_qcnn = [val_accuracy]
        

for epoch in range(num_epochs):
    # Training phase
    qcnn.train()
    running_correct = 0
    train_total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = qcnn(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses_qcnn.append(loss.item())

        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        train_total_samples += labels.size(0)

    train_accuracy = running_correct / train_total_samples
    train_accuracies_qcnn.append(train_accuracy)

    # Validation phase
    qcnn.eval()
    val_running_correct = 0
    val_total_samples = 0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = qcnn(val_images)

            _, val_preds = torch.max(val_outputs, 1)
            val_running_correct += (val_preds == val_labels).sum().item()
            val_total_samples += val_labels.size(0)

    val_accuracy = val_running_correct / val_total_samples
    val_accuracies_qcnn.append(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f} "
        f"Loss: {loss.item():.4f}"
    )


# In[] 
# ==== SAVE MODEL & PLOT RESULTS ====
train_losses = np.array(train_losses_qcnn)
train_accuracies = np.array(train_accuracies_qcnn)
val_accuracies = np.array(val_accuracies_qcnn)

# Save to logs 
model_description = f'C(k=2, s=2)-P(k=2)-D-L'
log_dir = f'logs/{dims[0]}x{dims[1]} {model_description} T={temperature} batch_size={batch_size} lr={lr}' 
os.mkdir(log_dir)

torch.save(qcnn.state_dict(), log_dir + '/qcnn_weights.pth')

np.savez_compressed(
    log_dir + '/history.npz',
    train_losses=train_losses,
    train_accuracies=train_accuracies,
    val_accuracies=val_accuracies,
)


# In[] 
# ==== PLOT LOSS CURVE ==== 
smoothed_loss = uniform_filter(train_losses_qcnn, size=len(train_loader) // 2)

# First subplot: Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(np.linspace(0, num_epochs, len(smoothed_loss)), smoothed_loss, linewidth=2)
plt.xlabel('Epoch')
plt.title('Loss (Smoothed)')

# Second subplot: Accuracy
ax2 = plt.subplot(1, 2, 2)
plt.title('Accuracy')
l1, = ax2.plot(range(num_epochs + 1), train_accuracies_qcnn, label='Training', linewidth=2)
l2, = ax2.plot(range(num_epochs + 1), val_accuracies_qcnn, label='Validation', linewidth=2)
ax2.set_xlabel('Epoch')
plt.xlim(-1, num_epochs)

# First legend: model type
legend1 = ax2.legend(handles=[l1, l2], loc='lower right')
ax2.add_artist(legend1)

# Second legend: line style (train/val)
train_line, = ax2.plot([], c='black', linewidth=2, label='Train')
val_line, = ax2.plot([], c='black', linewidth=2, linestyle='-.', label='Validation')

plt.tight_layout()
plt.show()
# %%
