from models.resnet import *
from utils import *

from sklearn.model_selection import train_test_split


from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.nn as nn
import torch.optim as optim

import argparse

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    default=".",
                    help="Directory containing params file")

if __name__ == "__main__":

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "PARAMS.JSON")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    train_root_dir = params.train_path
    train_df = datasets_to_df(params.train_path)

    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(train_df["label"])

    print("[INFO] y:", np.min(y), np.max(y))
    print("[INFO] Label Encoding:", labelEncoder.classes_)

    # split data
    X_train, X_val, y_train, y_val = train_test_split(train_df["fname"],
                                                        y,
                                                        test_size=params.SPLIT_RATIO,
                                                        stratify=y,
                                                        shuffle=True,
                                                        random_state=params.RANDOM_SEED)
    print(
        "[INFO] Training shape:",
        X_train.shape,
        y_train.shape,
        np.unique(y_train, return_counts=True)
    )

    print(
        "[INFO] Validation shape:",
        X_val.shape,
        y_val.shape,
        np.unique(y_val, return_counts=True)
    )

    cws = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)
    print("[INFO] Class weights:", cws)

    class_mapping = {k: v for k, v in enumerate(labelEncoder.classes_)}
    inv_class_mapping = {v: k for k, v in class_mapping.items()}


    print(f"----------To Numpy Array----------")
    #image_array = to_array(train_df["fname"])

    #image_array = image_array.reshape((-1, 3, 64, 64))

    print(f"----------Compute image mean & std----------")    
    #mean, std = get_stats_batch(image_array)

    image_array = None

    print(f"----------Load and Transform Images----------")
    train_transforms = get_train_transforms(params.HEIGHT,
                                      params.WIDTH,
                                      params.MEAN,
                                      params.STD)

    val_transforms = get_val_transforms(params.HEIGHT,
                                      params.WIDTH,
                                      params.MEAN,
                                      params.STD)

    train_dataset = ImagesDataset(X_train, y_train, None, train_transforms)
    val_dataset = ImagesDataset(X_val, y_val, None, val_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params.BATCH_SIZE,
                                               pin_memory=True,
                                               shuffle=True,
                                               num_workers=params.NUM_WORKERS)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=params.BATCH_SIZE,
                                             pin_memory=True,
                                             shuffle=False,
                                             num_workers=params.NUM_WORKERS)

    print(next(iter(train_loader)))
    im, lbl = next(iter(train_loader))
    print(im.shape, lbl.shape)

    print("[INFO] Training length:", len(train_loader.dataset))
    print("[INFO] Validation length:", len(val_loader.dataset))

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    device = get_device()
    print(f"----------Device type - {device}----------")

    
    # Set optimzer & lr_scheduler
    if params.NUM_CLASSES < 2:
        criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(cws)).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    model = ResNet18().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    #lr_finder(model, optimizer, criterion, train_loader, device, 1, 200)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, div_factor=10, final_div_factor=10,
    pct_start=10/params.EPOCHS,
    max_lr=params.LR,
    steps_per_epoch=len(train_loader),
    epochs=params.EPOCHS)


    print(f"----------Model Summary----------")
    get_summary(model, device)

    print(f"----------Training Model----------")
    results = train_model(model, criterion, device, train_loader, val_loader, optimizer, scheduler, params.EPOCHS)

    torch.save(model, params.SAVE_MODEL_PATH)

    print(f"----------Loss & Accuracy Plots----------")
    make_plot(results)


    '''max_images = 128
    test_images = [x[0] for x in testloader.dataset]
    test_images = torch.stack(test_images[:max_images])
    test_targets = torch.tensor(testloader.dataset.targets[:max_images]).to(device)
    print(f"----------Inference on {max_images} test images----------")


    test_predictions = model(test_images.to(device))

    print(f"----------Visualize model predictions----------")
    show_images_pred(test_images, test_targets, test_predictions, denorm)


    print(f"----------Visualize misclassified predictions----------")
    show_images_pred(test_images[miss_index], test_targets[miss_index], test_predictions[miss_index], denorm)'''
