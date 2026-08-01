from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from torchvision.datasets import MNIST
from torchvision import transforms

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision
import argparse
import torch_ema as ema
import tqdm

from models.DDPM import DDPM
from CAS_mnist_classifier import MNISTCLF
from utils import load_mnist_split, pick_device

import more_itertools as mit
import itertools as it

import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling synthesized MNISTDiffusion dataset")
    # argparse a list of strings:
    parser.add_argument("--run_names", type=str, nargs="+", default=[])

    args = parser.parse_args()

    print(args.run_names)

    return args


# train random forest classifier on MNIST


def as_image_batch(images):
    """Normalize any of the shapes floating around this script to (N, 1, img_size, img_size) on device."""
    return images.detach().reshape(-1, 1, img_size, img_size).float().to(device)


def train(clf, images, labels):
    clf.to(device)
    clf.train()

    optim = torch.optim.AdamW(clf.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    epo = 5

    images = as_image_batch(images)
    labels = labels.detach().reshape(-1).long().to(device)

    batch_size = 128
    n = len(images)
    for i in tqdm.tqdm(range(epo), desc="epoch"):

        epoch_total_loss = 0
        order = torch.randperm(n, device=device)
        batches = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]

            optim.zero_grad()
            preds = clf(images[idx])

            loss = criterion(preds, labels[idx])
            loss.backward()
            optim.step()
            epoch_total_loss += loss.item()
            batches += 1

        averaged = epoch_total_loss / max(batches, 1)
        print(f"\repoch {i+1:2}/{epo} loss: {averaged:.4f}", end="")
    print()
    return clf


def predict(clf, images):
    clf.eval()
    images = as_image_batch(images)
    out = []
    with torch.no_grad():
        for start in range(0, len(images), 1024):
            out.append(clf(images[start : start + 1024]).argmax(dim=1).cpu())
    return torch.cat(out).numpy()


def get_clf_avg_acc(images, labels, test_images, test_labels):

    accs = []
    for _ in range(5):

        clf = train(MNISTCLF(img_size), images, labels)
        acc, classwise_acc = get_classifier_performance(clf, test_images, test_labels)
        accs.append(np.array(classwise_acc))

    accs = np.array(accs)
    return accs.mean(axis=0), accs.std(axis=0)


def plot_classifier_performance(mean, sd, ax):

    # using bar plots where each bar is a class
    # bars should be side by side for each class
    # x-axis: class
    # y-axis: accuracy

    print(f"classifier {current_dataset} got CAS score: {mean.mean():.4f} +- {sd.mean():.4f}")

    width = 1 / (n_bars + 1)
    offset = current_bar_plot * width - 0.5

    ax.bar([i + offset for i in range(10)], mean, width=width, label="trained on {}".format(current_dataset))
    ax.errorbar([i + offset for i in range(10)], mean, yerr=sd, fmt="none", capsize=5, color="black")


def get_classifier_performance(real_clf, test_img, test_label):
    # create a plot showing classwise accuracy comparison

    real_pred = predict(real_clf, test_img)

    acc = accuracy_score(test_label.cpu(), real_pred)

    classwise_acc1 = []
    for i in range(10):
        classwise_acc1.append(
            accuracy_score(test_label.cpu()[test_label.cpu() == i].cpu(), real_pred[test_label.cpu() == i])
        )
    return acc, classwise_acc1


def visualize_images_and_labels(images, labels):
    # visualize some images with their labels
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(img_size, img_size), cmap="gray")
        plt.title(labels[i])
    plt.show()


if __name__ == "__main__":

    device = pick_device()
    print("Using device: {}".format(device))

    args = parse_args()

    img_size = 28

    # ? FIRST, TRAIN A CLASSIC MNIST CLASSIFIER ON THE REAL MNIST DATA AND GET THE VALIDATION-SET ACCURACY

    print("img_size: {}".format(img_size))

    train_size = 10000
    test_size = 10000

    # [0, 1] scaling, matching the domain the sampling scripts write their datasets in
    train_images_all, train_labels_all = load_mnist_split(True, img_size)
    real_train_images = train_images_all[:train_size].to(device)
    real_train_labels = train_labels_all[:train_size].to(device)

    validation_images_all, validation_labels_all = load_mnist_split(False, img_size)
    validation_images = validation_images_all[:test_size].to(device)
    validation_labels = validation_labels_all[:test_size].to(device)

    ax = plt.subplot(1, 1, 1)
    ax.set_xticks(range(10))
    ax.set_xlabel("class")
    ax.set_ylabel("accuracy")
    # set y bounds to [0, 1]
    ax.set_ylim(0, 1)

    n_bars = len(args.run_names) + 1
    current_bar_plot = 0
    current_dataset = "real"

    mean, sd = get_clf_avg_acc(real_train_images, real_train_labels, validation_images, validation_labels)
    plot_classifier_performance(mean, sd, ax)

    # ? THEN, FOR EVERY DDPM-GENERATED DATASET, TRAIN A CLASSIFIER ON THAT DATASET AND GET THE VALIDATION-SET ACCURACY

    for run_name in args.run_names:
        current_bar_plot += 1
        current_dataset = run_name

        print(f"Comparing classifier trained on real MNIST with classifier trained on {run_name}")
        # load synthesized dataset

        fake_train_data = (
            torch.from_numpy(np.load(f"synthesized/{run_name}/images.npy")).reshape(-1, 1, img_size, img_size).float(),
            torch.from_numpy(np.load(f"synthesized/{run_name}/labels.npy")).long(),
        )

        fake_train_images = fake_train_data[0].reshape(-1, img_size * img_size)[:train_size].to(device)
        fake_train_labels = fake_train_data[1][:train_size].to(device)

        mean, sd = get_clf_avg_acc(fake_train_images, fake_train_labels, validation_images, validation_labels)

        plot_classifier_performance(mean, sd, ax)

    print("final plotting")
    ax.legend()

    plt.savefig("classifier_comparison.png", dpi=300)
