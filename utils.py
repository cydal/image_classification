import torch
import torchvision
from torchsummary import summary
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
from PIL import Image

import glob2
import pandas as pd

from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import json
import os

train_losses = []
test_losses = []
train_acc = []
test_acc = []


class ImagesDataset(Dataset):
    """Image Classification Dataset"""
    def __init__(self, img_paths, labels=None, root_dir: str = None, transform=None):
        """
        Args:
            img_paths (pd.Series): Path to the images.
            labels (np.ndarray) : list or ndarray containing labels corresponding to images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_paths = img_paths
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        if self.root_dir:
          img_filename = os.path.join(self.root_dir, self.img_paths.iloc[idx])
        else:
          img_filename = self.img_paths.iloc[idx]

        img = np.array(Image.open(img_filename).convert("RGB"))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.labels is not None:
            return img, self.labels[idx]
        else:
            return img



class Params:
    """Load hyperparameters from a json file
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __str__(self) -> str:
        return str(self.__dict__)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def datasets_to_df(ds_path: str):
    """
    Convert folder path to DataFrame
    Args:
        ds_path (string): Path to dataset
    Returns:
        pd.DataFrame : A pandas dataframe containing paths to dataset and labels.
    """

    if not os.path.exists:
        raise FileNotFoundError(f"Directory Dataset not found: {ds_path}")

    filenames = glob2.glob(os.path.join(ds_path, "*/**.JPEG"))

    labels = []
    img_filenames = []

    for f in filenames:
      labels.append(f.split("/")[-2])
      img_filenames.append(f)

    df = pd.DataFrame({
        "fname": img_filenames,
        "label": labels
    })

    return(df)



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_summary(model, device):
  """
  Args:
      model (torch.nn Model): 
      device (str): cuda/CPU
  """
  print(summary(model, input_size=(3, 32, 32)))


def get_stats(images_array):
  """
  Args:
      images_array (numpy array): Image array
  Returns:
      mean: per channel mean
      std: per channel std
  """

  print('[Train]')
  print(' - Numpy Shape:', images_array.shape)
  #print(' - Tensor Shape:', images_array.shape)
  print(' - min:', np.min(images_array))
  print(' - max:', np.max(images_array))

  print('Divide by 255')
  images_array = images_array / 255.0

  mean = np.mean(images_array, axis=tuple(range(images_array.ndim-1)))
  std = np.std(images_array, axis=tuple(range(images_array.ndim-1)))

  print(f'\nDataset Mean - {mean}')
  print(f'Dataset Std - {std} ')

  return([mean, std])

def get_stats_batch(images_array):
  """
  Args:
      images_array (numpy array): Image array
  Returns:
      mean: per channel mean
      std: per channel std
  """
  batch = 100
  steps = images_array.shape[0] // 100

  print('[Train]')
  print(' - Numpy Shape:', images_array.shape)
  #print(' - Tensor Shape:', images_array.shape)
  print(' - min:', np.min(images_array))
  print(' - max:', np.max(images_array))

  print('Divide by 255')
  images_array = images_array / 255.0

  std, mean = [], []

  print('Computing mean & std')
  for i in range(steps):
    mean.append(np.mean(images_array[i*batch:batch+(i*batch)], axis=tuple(range(images_array.ndim-1))))
    std.append(np.std(images_array[i*batch:batch+(i*batch)], axis=tuple(range(images_array.ndim-1))))

  mean, std = np.array(mean).mean(axis=0), np.array(std).mean(axis=0)

  print(f'\nDataset Mean - {mean}')
  print(f'Dataset Std - {std} ')

  return([mean, std])


def get_train_transform(MEAN, STD):

    train_transform = A.Compose([
                                A.Cutout(max_h_size=16, max_w_size=16),
                                A.Normalize(mean=(MEAN), 
                                            std=STD),
                                ToTensorV2(),
    ])
    return(train_transform)


def get_test_transform(MEAN, STD):

    test_transform = A.Compose([
                                A.Normalize(mean=MEAN, 
                                            std=STD),
                                ToTensorV2(),
    ])
    return(test_transform)


def get_train_transforms(h, w, mu, std):
    """
    Args:
        h (int): image height
        w (int):image width
        mu (array): image mean
        std (array): standard deviation
    Returns:
        train_transforms (Albumentation): Transform Object
    """

    train_transform = A.Compose([
                            A.Resize(h, w, cv2.INTER_NEAREST),
                            A.CenterCrop(h, w),
                            A.Cutout(max_h_size=16, max_w_size=16),
                            A.Normalize(mean=(mu), 
                                        std=std),
                            ToTensorV2()
    ])

    return(train_transform)

def get_val_transforms(h, w, mu, std):
    """
    Args:
        h (int): image height
        w (int):image width
        mu (array): image mean
        std (array): standard deviation
    Returns:
        val_transforms (Albumentation): Transform Object
    """
    val_transforms = A.Compose([
                            A.Resize(h, w, cv2.INTER_NEAREST),
                            A.CenterCrop(h, w),
                            A.Normalize(mean=(mu), 
                                        std=std),
                            ToTensorV2()
    ])

    return(val_transforms)


def get_device():
  """
  Returns:
      device (str): device type
  """
  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # For reproducibility
  if cuda:
      torch.cuda.manual_seed(SEED)
  else:
    torch.manual_seed(SEED)

  return(device)



def train(model, device, criterion, train_loader, optimizer, epoch):
  """
  Args:
      model (torch.nn Model): 
      device (str): device type
      criterion (criterion) - Loss Function
      train_loader (DataLoader) - DataLoader Object
      optimizer (optimizer) - Optimizer Object
      epoch (int) - Number of epochs
  """
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, criterion, test_loader):
  """
  Args:
      model (torch.nn Model): 
      device (str): device type
      criterion (criterion) - Loss Function
      test_loader (DataLoader) - DataLoader Object
  """
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          test_loss += criterion(output, target).item()  # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
  
  test_acc.append(100. * correct / len(test_loader.dataset))



def train_model(model, criterion, device, train_loader, test_loader, optimizer, scheduler, EPOCHS):
  """
  Args:
      model (torch.nn Model): 
      criterion (criterion) - Loss Function
      device (str): cuda/CPU
      train_loader (DataLoader) - DataLoader Object
      optimizer (optimizer) - Optimizer Object
      scheduler (scheduler) - scheduler object
      EPOCHS (int) - Number of epochs
  Returns:
      results (list): Train/test - Accuracy/Loss 
  """
  for epoch in range(EPOCHS):
      print("EPOCH:", epoch)
      train(model, device, criterion, train_loader, optimizer, epoch)
      scheduler.step()
      test(model, device, criterion, test_loader)

  results = [train_losses, test_losses, train_acc, test_acc]
  return(results)


def make_plot(results):
    """
    Args:
        images (list of list): Loss & Accuracy List
    """
    tr_losses = results[0]
    te_losses = results[1]
    tr_acc = results[2]
    te_acc = results[3]


    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(tr_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(tr_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(te_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(te_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()


def to_array(f_name):
    """
    Args:
        f_name (Pandas Series - str): image file names
    Returns:
        images_array (numpy array): images array
    """
    images_array = np.array([np.array(Image.open(x).convert("RGB")) for x in f_name])
    return(images_array)