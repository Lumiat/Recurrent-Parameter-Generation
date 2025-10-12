# set global seed
import random
import numpy as np
import torch
import sys, os
sys.path.append(os.path.dirname(__file__))

seed = SEED = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

try:  # relative import
    from model import Model
except ImportError:
    from .model import Model

# import
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import SVHN as Dataset
from tqdm.auto import tqdm
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# load additional config
# -----------------------------
config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(config_file, "r") as f:
    additional_config = json.load(f)

# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "dataset_root": "from_additional_config",
    "batch_size": 500 if __name__ == "__main__" else 200,
    "num_workers": 32,
    "learning_rate": 3e-3,
    "weight_decay": 0.1,
    "epochs": 50,
    "save_learning_rate": 1e-5,
    "total_save_number": 50,
    "tag": os.path.basename(os.path.dirname(__file__)),
}
config.update(additional_config)

# -----------------------------
# Data
# -----------------------------
transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# use split='train' and split='test'
train_dataset = Dataset(
    root=config["dataset_root"],
    split='train',
    download=True,
    transform=transform_train,
)
test_dataset = Dataset(
    root=config["dataset_root"],
    split='test',
    download=True,
    transform=transform_test,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    pin_memory_device="cuda",
)

# -----------------------------
# Model
# -----------------------------
model, head = Model()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

# -----------------------------
# Training
# -----------------------------
def train(model=model, optimizer=optimizer, scheduler=scheduler):
    model.train()
    for epoch in range(config["epochs"]):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader),
                                                 total=len(train_loader),
                                                 desc=f"Epoch {epoch+1}/{config['epochs']}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")
        test(model)

# -----------------------------
# Test
# -----------------------------
@torch.no_grad()
def test(model=model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader),
                                             total=len(test_loader),
                                             desc="Testing"):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = correct / total
    avg_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")
    model.train()
    return avg_loss, acc

# -----------------------------
# Save training checkpoints
# -----------------------------
def save_train(model=model, optimizer=optimizer):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % (len(train_dataset) // train_loader.batch_size // config["total_save_number"]) == 0:
            _, acc = test(model)
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_state = {k: v.cpu().to(torch.float32) for k, v in model.state_dict().items()}
            save_path = f"checkpoint/{str(batch_idx).zfill(4)}_acc{acc:.4f}_seed{seed:04d}_{config['tag']}.pth"
            torch.save(save_state, save_path)
            print("Checkpoint saved:", save_path)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    test(model=model)
    train(model=model, optimizer=optimizer, scheduler=scheduler)
    save_train(model=model, optimizer=optimizer)
