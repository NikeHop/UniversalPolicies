import io

import h5py
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import torch

from robonet.datasets import load_metadata
from torchvision.io import write_video


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

    raise NotImplementedError

    return images


all_robonet = load_metadata("../../../data/hdf5")
print("Metadata loaded")
no_sawyer_data = all_robonet[all_robonet["robot"] != "sawyer"]
no_sawyer_files = no_sawyer_data.get_shuffled_files()
print("Files loaded")
print(all_robonet.keys())


def as_gif(images, path="temp.gif"):
    # Render the images as the gif:
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000, loop=0)
    gif_bytes = open(path, "rb").read()
    return gif_bytes



annotations_values = []
for file in no_sawyer_files:

    metadata = no_sawyer_data.get_file_metadata(file)
    annotations_values.append(metadata["contains_annotation"])

    if metadata["contains_annotation"]:
        print(metadata)
        print(metadata["contains_annotation"])
        
        with h5py.File(file, "r") as f:
            print(f.keys())
            print(f["misc"].keys())
        break

    """
    print(metadata)
    with h5py.File(file, 'r') as f:
        images = load_camera_imgs(0, f, metadata)
        images = images.copy()
        print(f"Images shape: {images.shape}")  
        images = torch.from_numpy(images)
        write_video("robonet.mp4", images, fps=5)
    break
    """

print(list(set(annotations_values)))
