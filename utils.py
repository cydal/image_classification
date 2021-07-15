import torch
import torchvision
from torchsummary import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2


from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim

import numpy as np

train_losses = []
test_losses = []
train_acc = []
test_acc = []



classes = ["airplane", "automobile", "bird", "cat", "deer", 
           "dog", "frog", "horse", "ship", "truck"]



class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label



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


def get_train_transform(MEAN, STD, PAD=4):

    train_transform = A.Compose([
                                A.PadIfNeeded(min_height=32+PAD, 
                                            min_width=32+PAD, 
                                            border_mode=cv2.BORDER_CONSTANT,
                                            value=(MEAN)),
                                A.RandomCrop(32, 32),
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




def get_summary(model, device):
  """
  Args:
      model (torch.nn Model): Original data with no preprocessing
      device (str): cuda/CPU
  """
  print(summary(model, input_size=(3, 32, 32)))


def get_stats(trainloader):
  """
  Args:
      trainloader (trainloader): Original data with no preprocessing
  Returns:
      mean: per channel mean
      std: per channel std
  """
  train_data = trainloader.dataset.data

  print('[Train]')
  print(' - Numpy Shape:', train_data.shape)
  print(' - Tensor Shape:', train_data.shape)
  print(' - min:', np.min(train_data))
  print(' - max:', np.max(train_data))

  train_data = train_data / 255.0

  mean = np.mean(train_data, axis=tuple(range(train_data.ndim-1)))
  std = np.std(train_data, axis=tuple(range(train_data.ndim-1)))

  print(f'\nDataset Mean - {mean}')
  print(f'Dataset Std - {std} ')

  return([mean, std])


def get_train_loader(transform=None):
  """
  Args:
      transform (transform): Albumentations transform
  Returns:
      trainloader: DataLoader Object
  """
  if transform:
    trainset = Cifar10SearchDataset(transform=transform)
  else:
    trainset = Cifar10SearchDataset(root="~/data/cifar10", train=True, 
                                    download=True)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)
  return(trainloader)


def get_test_loader(transform=None):
  """
  Args:
      transform (transform): Albumentations transform
  Returns:
      testloader: DataLoader Object
  """
  if transform:
    testset = Cifar10SearchDataset(transform=transform, train=False)
  else:
    testset = Cifar10SearchDataset(train=False)
  testloader = torch.utils.data.DataLoader(testset, batch_size=128, 
                                         shuffle=False, num_workers=2)

  return(testloader)


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


def superimpose(heatmap, img, denorm):
  img = np.transpose(denorm(img.cpu()), (1, 2, 0))
  heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
  heatmap = np.uint8(255*heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  superimposed_img = (heatmap * 0.4) + 255*img.numpy()
  return((heatmap, superimposed_img/superimposed_img.max()))


def train(model, device, criterion, train_loader, optimizer, epoch):
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
      model (torch.nn Model): Original data with no preprocessing
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



def get_idxs(results, test_targets, device):
  """
  Args:
      results (tensor): predictions
      test_targets (tensor): Ground truth labels

  Returns:
      miss_index: index of misclassifier images
      hit_index: index of correctly classifier images
  """
  miss_index = torch.where((results.argmax(dim=1) == torch.tensor(test_targets).to(device)) == False)[0]
  hit_index = torch.where((results.argmax(dim=1) == torch.tensor(test_targets).to(device)) == True)[0]
  return((miss_index, hit_index))


def show_images_pred(images, targets, preds, denorm):
  """
  Args:
      images (tensor): images array
      targets (tensor): Ground truth labels
      preds (tensor): Predictions
  """
  plt.figure(figsize=(15 ,15))
  for i in range(16):
      plt.subplot(4, 4,i+1)
      label = classes[targets[i].cpu()]
      pred = classes[preds[i].cpu().argmax()]
      plt.title(f"(T)-{label} - (P)-{pred}")
      plt.imshow(np.transpose(denorm(images[i].cpu()), (1, 2, 0)))
  plt.show()


def gradcam_heatmap(model, results, test_images, device):
  """
  Args:
      model (torch.nn): Torch model
      test_targets (tensor): Ground truth labels
      test_images (tensor): images array
      device (str): Device type

  Returns:
      heatmaps (tensor): heatmaps array
  """
  results[torch.arange(len(results)), 
          results.argmax(dim=1)].backward(torch.ones_like(results.argmax(dim=1)))

  gradients = model.get_activations_gradient()

  pooled_gradients = torch.mean(gradients, dim=[2, 3])


  activations = model.get_activations(test_images.to(device)).detach()

  # weight the channels by corresponding gradients
  for j in range(activations.shape[0]):
    for i in range(512):
        activations[j, i, :, :] *= pooled_gradients[j, i]
      

  # average the channels of the activations
  heatmaps = torch.mean(activations, dim=1).squeeze()


  # relu on top of the heatmap
  heatmaps = np.maximum(heatmaps.cpu(), 0)

  # normalize the heatmap
  heatmaps /= torch.max(heatmaps)

  return(heatmaps)


def superimpose(heatmap, img, denorm):
  """
  Args:
      heatmap (tensor): Gradient heatmap
      img (tensor): Image array

  Returns:
      superimposed_img (numpy): image array
  """
  img = np.transpose(denorm(img.cpu()), (1, 2, 0))
  heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
  heatmap = np.uint8(255*heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  superimposed_img = (heatmap * 0.4) + 255*img.numpy()
  
  return(superimposed_img/superimposed_img.max())

def show_images_cam(images, targets, preds, heatmaps, idx, denorm):
  """
  Args:
      images (tensor): Images array
      targets (tensor): Ground truth labels
      preds (tensor): Predictions
      heatmaps (tensor): Gradient heatmaps
      idx (tensor): Subset index

  Returns:
      heatmaps (tensor): heatmaps array
  """
  images, targets = images[idx], targets[idx]
  preds, heatmaps = preds[idx], heatmaps[idx]

  plt.figure(figsize=(15 ,15))
  for i in range(16):
      plt.subplot(4, 4,i+1)
      label = classes[targets[i].cpu()]
      pred = classes[preds[i].cpu().argmax()]
      plt.title(f"(T)-{label} - (P)-{pred}")
      plt.imshow(superimpose(heatmaps[i], images[i], denorm))
      
  plt.show()


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