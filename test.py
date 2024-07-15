import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.models_v1 import LSTMClassifier
from src.utils import set_seed

train_set = ThingsMEGDataset("train", "data")
print(type(train_set))