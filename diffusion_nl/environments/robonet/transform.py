import argparse
import os

import yaml
import h5py

import cv2
import io
import imageio
import numpy as np
import torch
import tqdm

from torchvision.transforms import InterpolationMode, Resize
from robonet.datasets import load_metadata


def load_camera_imgs(
    cam_index, file_pointer, file_metadata, target_dims=None, start_time=0, n_load=None
):
    cam_group = file_pointer["env"]["cam{}_video".format(cam_index)]
    old_dims = file_metadata["frame_dim"]
    length = file_metadata["img_T"]
    encoding = file_metadata["img_encoding"]
    image_format = file_metadata["image_format"]

    if n_load is None:
        n_load = length

    old_height, old_width = old_dims
    if target_dims is not None:
        target_height, target_width = target_dims
    else:
        target_height, target_width = old_height, old_width

    resize_method = cv2.INTER_CUBIC
    if target_height * target_width < old_height * old_width:
        resize_method = cv2.INTER_AREA

    images = np.zeros((n_load, target_height, target_width, 3), dtype=np.uint8)
    if encoding == "mp4":
        buf = io.BytesIO(cam_group["frames"][:].tostring())
        img_buffer = [
            img
            for t, img in enumerate(imageio.get_reader(buf, format="mp4"))
            if start_time <= t < n_load + start_time
        ]
    elif encoding == "jpg":
        img_buffer = [
            cv2.imdecode(cam_group["frame{}".format(t)][:], cv2.IMREAD_COLOR)[
                :, :, ::-1
            ]
            for t in range(start_time, start_time + n_load)
        ]
    else:
        raise ValueError("encoding not supported")

    for t, img in enumerate(img_buffer):
        if (old_height, old_width) == (target_height, target_width):
            images[t] = img
        else:
            images[t] = cv2.resize(
                img, (target_width, target_height), interpolation=resize_method
            )

    if image_format == "RGB":
        return images
    elif image_format == "BGR":
        return images[:, :, :, ::-1]

    raise NotImplementedError()


def transform_data_autoencoder(config):
    """
    Transforms the data into the format necessary for autoencoder training
    """
    # Setting directory
    training_directory = os.path.join(config["save_dir"], "training")
    validation_directory = os.path.join(config["save_dir"], "validation")

    # Load metadata
    robonet_metadata = load_metadata("../../../data/hdf5")

    if config["filter_sawyer"]:
        # Filter files that contain sawyer data
        robonet_metadata = robonet_metadata[robonet_metadata["robot"] != "sawyer"]

    files = robonet_metadata.get_shuffled_files()
    n_files = len(files)
    training_files = files[: int(config["ratio"] * n_files)]
    validation_files = files[int(config["ratio"] * n_files) :]
    total_frames = 0
    batch_count = 0
    image_batch = []

    
    for file in tqdm.tqdm(training_files):
        with h5py.File(file, "r") as file_pointer:
            # Load metadata
            file_metadata = robonet_metadata.get_file_metadata(file)

            # Load images
            images = load_camera_imgs(0, file_pointer, file_metadata, target_dims=(128, 128))
            print(images.shape)

        # Save images to disk
        for img in images:
            image_batch.append(img)

            if len(image_batch) == 10:
                imgs = np.stack(image_batch, axis=0)
                print(f"Saved batch {imgs.shape}")
                image_batch = []
                total_frames += 10
                np.savez(
                    os.path.join(training_directory, f"robonet_{batch_count}.npz"), imgs
                )
                batch_count += 1
    
    batch_count = 0
    image_batch = []
    for file in tqdm.tqdm(validation_files):
        with h5py.File(file, "r") as file_pointer:
            # Load metadata
            file_metadata = robonet_metadata.get_file_metadata(file)

            # Load images
            images = load_camera_imgs(0, file_pointer, file_metadata, target_dims=(128, 128))
            print(images.shape)

        # Save images to disk
        for img in images:
            image_batch.append(img)

            if len(image_batch) == 10:
                imgs = np.stack(image_batch, axis=0)
                image_batch = []
                print(imgs.shape)
                total_frames += 10
                np.savez(
                    os.path.join(validation_directory, f"robonet_{batch_count}.npz"),
                    imgs,
                )
                batch_count += 1

    print(f"Total frames Added: {total_frames}")


if __name__ == "__main__":
    config = {"ratio":0.9, "save_dir":"../../../data/autoencoder", "filter_sawyer":True}

    transform_data_autoencoder(config)
