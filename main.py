from models.resnet import *
from utils import *


from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.nn as nn
import torch.optim as optim

import argparse

from sklearn.preprocessing import LabelEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    default=".",
                    help="Directory containing params file")

if __name__ == "__main__":

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    train_root_dir = params.train_path
    train_df = datasets_to_df(params.train_path)

    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(train_df["label"])
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



    print(f"----------Compute image mean & std----------")    
    mean, std = get_stats(trainloader)


    denorm = UnNormalize(mean, std) 

    print(f"----------Normailizing Images----------")
    train_transform = get_train_transform(mean, std)
    test_transform = get_test_transform(mean, std)

    print(f"----------Load and Transform Images----------")
    trainloader = get_train_loader(transform=train_transform)
    testloader = get_test_loader(transform=test_transform)


    device = get_device()
    print(f"----------Device type - {device}----------")


    print(f"----------Model Summary----------")
    model = ResNet18().to(device)
    get_summary(model, device)


    model =  ResNet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = OneCycleLR(optimizer, max_lr=0.05, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    print(f"----------Training Model----------")
    results = train_model(model, criterion, device, trainloader, testloader, optimizer, scheduler, EPOCHS)

    print(f"----------Loss & Accuracy Plots----------")
    make_plot(results)


    max_images = 128
    test_images = [x[0] for x in testloader.dataset]
    test_images = torch.stack(test_images[:max_images])
    test_targets = torch.tensor(testloader.dataset.targets[:max_images]).to(device)
    print(f"----------Inference on {max_images} test images----------")


    test_predictions = model(test_images.to(device))

    miss_index, hit_index = get_idxs(test_predictions, test_targets, device)


    print(f"----------missclassifid index length is {len(miss_index)}----------")
    print(f"----------Correctly classified index length is {len(hit_index)}----------")


    print(f"----------Visualize model predictions----------")
    show_images_pred(test_images, test_targets, test_predictions, denorm)


    print(f"----------Visualize misclassified predictions----------")
    show_images_pred(test_images[miss_index], test_targets[miss_index], test_predictions[miss_index], denorm)


    print(f"----------Generate heatmaps for test images----------")
    heatmaps = gradcam_heatmap(model, test_predictions, test_images, device)


    hit_maps = heatmaps[hit_index]
    miss_maps = heatmaps[miss_index]

    print(f"----------isualize misclassified GRADCAM----------")
    show_images_cam(test_images, test_targets, test_predictions, heatmaps, miss_index, denorm)


    print(f"----------Visualize correctly classified GRADCAM----------")
    show_images_cam(test_images, test_targets, test_predictions, heatmaps, hit_index, denorm)