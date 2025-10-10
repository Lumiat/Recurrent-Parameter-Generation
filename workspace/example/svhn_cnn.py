import sys, os, json
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation")+1])
sys.path.append(root)
os.chdir(root)
with open("./workspace/config.json", "r") as f:
    additional_config = json.load(f)
USE_WANDB = additional_config["use_wandb"]

# set global seed
import random
import numpy as np
import torch
seed = SEED = 999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# other
import math
import random
import warnings
from _thread import start_new_thread
warnings.filterwarnings("ignore", category=UserWarning)
if USE_WANDB: import wandb
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.cuda.amp import autocast
# model
from model import MambaDiffusion as Model
from model.diffusion import DDPMSampler, DDIMSampler
from model.fusion_mlp import FusionMLP
from model.feature_extractor import ImageFeatureExtractor, TextFeatureExtractor

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import AutocastKwargs
from accelerate import Accelerator
# dataset
from dataset import SVHN_CNN as Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms




config = {
    "seed": SEED,
    # dataset setting
    "dataset": Dataset,
    "dim_per_token": 8192,
    "sequence_length": 'auto',
    # feature extraction setting
    "text_extractor_backbone": 'ViT-B/32',
    "image_extractor_backbone": 'resnet50',
    "description": 
        'SVHN dataset consists of 32x32 pixel images of house numbers captured from Google Street View. \
        Each image contains a single digit (0-9), similar to MNIST, \
        though some images may have distractors at the edges. \
        The dataset has 10 classes, corresponding to the digits 0 through 9.',
    "class_names": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9"
    ],
    "class_image_paths": '',
    "dist_image_path":'',
    "combine": 'concat', # concat, mlp, weighted_sum
    # train setting
    "batch_size": 8,
    "num_workers": 16,
    "total_steps": 80000,
    "learning_rate": 0.00003,
    "weight_decay": 0.0,
    "save_every": 80000//30,
    "print_every": 50,
    "autocast": lambda i: 5000 < i < 45000,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_batch_size": 1,  # fixed, don't change this
    "generated_path": Dataset.generated_path,
    "test_command": Dataset.test_command,
    # to log
    "model_config": {
        "num_permutation": 'auto',
        # mamba config
        "d_condition": 1,
        "d_model": 8192,
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 2,
        # diffusion config
        "diffusion_batch": 512,
        "layer_channels": [1, 32, 64, 128, 64, 32, 1],
        "model_dim": "auto",
        "condition_dim": "auto",
        "kernel_size": 7,
        "sample_mode": DDPMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
        "forward_once": True,
    },
    "tag": "test_svhn_cnn",
}




# Data
print('==> Preparing data..')
train_set = config["dataset"](dim_per_token=config["dim_per_token"])
print("Dataset length:", train_set.real_length)
print("input shape:", train_set[0][0].shape)
if config["model_config"]["num_permutation"] == "auto":
    config["model_config"]["num_permutation"] = train_set.max_permutation_state
if config["model_config"]["condition_dim"] == "auto":
    config["model_config"]["condition_dim"] = config["model_config"]["d_model"]
if config["model_config"]["model_dim"] == "auto":
    config["model_config"]["model_dim"] = config["dim_per_token"]
if config["sequence_length"] == "auto":
    config["sequence_length"] = train_set.sequence_length
    print(f"sequence length: {config['sequence_length']}")
else:  # set fixed sequence_length
    assert train_set.sequence_length == config["sequence_length"], f"sequence_length={train_set.sequence_length}"
train_loader = DataLoader(
    dataset=train_set,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    persistent_workers=True,
    drop_last=True,
    shuffle=True,
)

# Model
print('==> Building model..')
Model.config = config["model_config"]
model = Model(
    sequence_length=config["sequence_length"],
    positional_embedding=train_set.get_position_embedding(
        positional_embedding_dim=config["model_config"]["d_model"]
    )  # positional_embedding
)  # model setting is in model

# Optimizer
print('==> Building optimizer..')
optimizer = optim.AdamW(
    params=model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=config["total_steps"],
)

# accelerator
if __name__ == "__main__":
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs,])
    if config["dim_per_token"] > 12288 and accelerator.state.num_processes == 1:
        print(f"\033[91mWARNING: With token size {config['dim_per_token']}, we suggest to train on multiple GPUs.\033[0m")
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)


# wandb
if __name__ == "__main__" and USE_WANDB and accelerator.is_main_process:
    wandb.login(key=additional_config["wandb_api_key"])
    wandb.init(project="Recurrent-Parameter-Generation", name=config['tag'], config=config,)

# Feature Extraction
print("==> Extracting Feature..")
def extract_feature(description, class_names, class_image_paths, dist_image_path):
    """
    :param description: str, dataset description
    :param class_names: list[str], each class name (e.g. ["airplane", "automobile", ...])
    :param class_image_paths: list[str], path to one sample image per class
    :param dist_image_path: str, path to bar chart showing dataset distribution
    """
    assert len(class_names) == len(class_image_paths), \
        "class_names and class_image_paths must have the same length!"

    # === CLIP part (text + class images) ===
    text_extractor = TextFeatureExtractor(config["text_extractor_backbone"])

    # 1) extract text features for class names
    class_text_features = text_extractor.extract(class_names)  # shape [num_classes, dim]

    # 2) extract image features for class images
    image_feature_list = []
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    for img_path in class_image_paths:
        image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(text_extractor.device)
        with torch.no_grad():
            img_feature = text_extractor.model.encode_image(image_tensor)
            img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
        image_feature_list.append(img_feature.squeeze(0))
    class_image_features = torch.stack(image_feature_list, dim=0)  # shape [num_classes, dim]

    # 3) fuse each (text, image) pair
    # simple concat → [num_classes, 2*dim]
    class_pair_features = torch.cat([class_text_features, class_image_features], dim=-1)

    # 4) also include the dataset description as a global feature
    dataset_text_feature = text_extractor.extract(description)  # shape [dim]
    dataset_text_feature = dataset_text_feature.unsqueeze(0)   # [1, dim]

    # final clip_feature = concat(global description, per-class pairs)
    clip_feature = torch.cat([dataset_text_feature, class_pair_features.flatten().unsqueeze(0)], dim=-1)
    # shape: [1, dim + num_classes*2*dim]

    # === Distribution feature part (ResNet) ===
    dist_feature_extractor = ImageFeatureExtractor(config["image_extractor_backbone"])
    dist_feature = dist_feature_extractor.extract(dist_image_path)  # shape [2048] for resnet50

    return clip_feature, dist_feature


def combine_feature(clip_feature, dist_feature):
    """
    :param clip_feature: torch.Tensor, shape [1, dim_clip]
    :param dist_feature: torch.Tensor, shape [1, dim_dist]
    :return: torch.Tensor, shape [1, d_condition]
    """
    clip_dim = clip_feature.shape[-1]
    dist_dim = dist_feature.shape[-1]
    total_dim = clip_dim + dist_dim

    if config["combine"] == "concat":
        feature = torch.cat([clip_feature, dist_feature], dim=-1)

    elif config["combine"] == "mlp":
        fusion_mlp = FusionMLP(
            input_dim=total_dim,
            hidden_dim=512,
            output_dim=config["model_config"]["d_condition"]
        ).to(clip_feature.device)
        feature = fusion_mlp(torch.cat([clip_feature, dist_feature], dim=-1))
        return feature  # already projected to d_condition

    elif config["combine"] == "weighted_sum":
        # learnable weights
        alpha = nn.Parameter(torch.tensor(0.5, device=clip_feature.device))
        beta = 1 - alpha
        # first project each feature into same dim
        proj_clip = nn.Linear(clip_dim, config["model_config"]["d_condition"]).to(clip_feature.device)
        proj_dist = nn.Linear(dist_dim, config["model_config"]["d_condition"]).to(clip_feature.device)
        feature = alpha * proj_clip(clip_feature) + beta * proj_dist(dist_feature)
        return feature  # already d_condition

    else:
        raise ValueError(f"Unknown combine mode: {config['combine']}")

    # For concat case: project to d_condition
    projector = nn.Linear(total_dim, config["model_config"]["d_condition"]).to(feature.device)
    feature = projector(feature)
    return feature


# Training
print('==> Defining training..')
def train():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> Start training..")
    model.train()
    for batch_idx, (param, permutation_state) in enumerate(train_loader):
        optimizer.zero_grad()
        # train
        # noinspection PyArgumentList
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=config["autocast"](batch_idx))):
            clip_feature, dist_feature = extract_feature(config["description"], config["class_names"], config["class_image_paths"], config["dist_image_path"])
            condition = combine_feature(clip_feature, dist_feature)
            loss = model(output_shape=param.shape, x_0=param, condition=condition, permutation_state=permutation_state)
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.is_main_process:
            scheduler.step()
        # to logging losses and print and save
        if USE_WANDB and accelerator.is_main_process:
            wandb.log({"train_loss": loss.item()})
        elif USE_WANDB:
            pass  # don't print
        else:  # not use wandb
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                print('Loss: %.6f' % (train_loss/this_steps))
                this_steps = 0
                train_loss = 0
        if batch_idx % config["save_every"] == 0 and accelerator.is_main_process:
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = accelerator.unwrap_model(model).state_dict()
            torch.save(state, os.path.join(config["checkpoint_save_path"], config["tag"]+".pth"))
            generate(save_path=config["generated_path"], need_test=True)
        if batch_idx >= config["total_steps"]:
            break


def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.no_grad():
        prediction = model(sample=True)
        generated_norm = prediction.abs().mean()
    print("Generated_norm:", generated_norm.item())
    if USE_WANDB:
        wandb.log({"generated_norm": generated_norm.item()})
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        start_new_thread(os.system, (config["test_command"],))
    model.train()
    return prediction




if __name__ == '__main__':
    train()
    del train_loader  # deal problems by dataloader
    print("Finished Training!")
    exit(0)