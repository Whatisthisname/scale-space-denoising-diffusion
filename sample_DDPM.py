# load DDPM from checkpoint.
import glob
import re

import torch
from models.DDPM import DDPM
import argparse
import torchvision

from utils import denormalize_images, pick_device

# load latest checkpoint:


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling synthesized MNISTDiffusion dataset")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--rescale", type=int, default=0)
    parser.add_argument("-o", "--output", type=str, default="_")
    parser.add_argument("--stack_samples", action="store_true", default=False)
    parser.add_argument("--compute_speed", action="store_true", default=False)

    args = parser.parse_args()

    # load run_name/args.txt file and parse:
    args_file = "checkpoints/{}/args.txt".format(args.run_name)
    argsDict = {}
    with open(args_file, "r") as f:
        for l in f.readlines():
            k, v = l.split()
            k = k.replace("--", "", 1)  # remove first occurence of "--"
            value = v
            try:
                value = eval(v)
            except:
                pass
            argsDict[k] = value

    # overwrite args with args from args.txt:
    for k, v in argsDict.items():
        setattr(args, k, v)

    return args


import glob
import re
import os
import shutil

# import cv2
import numpy as np


def main(args):

    device = pick_device()
    print("Sampling on {}".format(device))
    model = DDPM(
        args.img_size,
        ctx_sz=1 + 10,
        markov_states=args.markov_states,
        unet_stages=args.unet_stages,
        noise_schedule_param=args.noise_power,
        device=device,
    ).to(device)

    checkpoints = glob.glob("checkpoints/{}/*.pth".format(args.run_name))
    if len(checkpoints) == 0:
        print(f"No checkpoints found with run name '{args.run_name}', aborting")
        exit()
    else:
        checkpoints.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]))
        latest_checkpoint = checkpoints[-1]
        print("Loading checkpoint: {}".format(latest_checkpoint))
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))

    model.to(device)
    # running statistics, not batch statistics: otherwise a sample depends on its batch mates
    model.eval()
    print("model loaded.")

    # sample from model:
    fname = args.run_name if args.output == "_" else args.output

    samples = denormalize_images(model.sample(30, torch.arange(30) % 10, keep_intermediate=False), args.img_size)

    torchvision.utils.save_image(
        samples,
        "previews/" + fname + str(torch.rand((1,), dtype=torch.float16).item())[:4] + ".png",
        nrow=10,
        padding=6,
    )

    # create output directory:
    os.makedirs("synthesized/{}".format(fname), exist_ok=True)
    if not args.stack_samples:
        # delete previous samples, but not the folder:
        try:
            os.remove("synthesized/{}/{}".format(fname, "images.npy"))
            os.remove("synthesized/{}/{}".format(fname, "labels.npy"))
        except FileNotFoundError as e:
            pass

    batch_size = 64

    images = []
    labels = []

    img_size = args.img_size if args.rescale == 0 else args.rescale

    with torch.no_grad():
        for i in range(args.size // batch_size):
            # print(f"Sampling batch {i+1} / {args.size // batch_size}")

            gen_labels = torch.randint(0, 10, (batch_size,)).to(device).tolist()

            samples = model.sample(batch_size, target_label=gen_labels, keep_intermediate=False)

            # back to the [0, 1] image domain before any resizing, so the saved dataset is
            # directly comparable to the real MNIST the CAS classifiers validate against
            samples = denormalize_images(samples, args.img_size)

            if args.rescale != 0:
                samples = torchvision.transforms.Resize(args.rescale, antialias=True)(samples)

            samples = samples.cpu().numpy()

            images.append(samples)
            labels.append(gen_labels)

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    if args.stack_samples:
        prev_images, prev_labels = (
            torch.from_numpy(np.load(f"synthesized/{fname}/images.npy")).reshape(-1, 1, img_size, img_size).float(),
            torch.from_numpy(np.load(f"synthesized/{fname}/labels.npy")).long(),
        )

        images = np.concatenate([prev_images, images], axis=0)
        labels = np.concatenate([prev_labels, labels], axis=0)

    np.save(f"synthesized/{fname}/images.npy", images)
    np.save(f"synthesized/{fname}/labels.npy", labels)

    if args.compute_speed:

        n = 100

        print(f"sampling {n} images to see how fast it goes:")
        import time

        start = time.time()
        for i in range(n // 2):
            gen_labels = torch.randint(0, 10, (2,)).tolist()
            samples = model.sample(2, target_label=gen_labels, keep_intermediate=False)

        end = time.time()
        print(f"{n} samples took {end - start} seconds, which is an average of {(end - start) / n} seconds per sample")


if __name__ == "__main__":
    args = parse_args()
    main(args)
