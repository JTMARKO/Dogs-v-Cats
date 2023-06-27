import tensorflow as tf
import tarfile
import numpy as np
import os
from pathlib import Path

import PIL
import PIL.Image
import matplotlib.pyplot as plt


tar_file_path = 'annotations.tar.gz'
extracted_folder_path = 'data'

def extract_tar(tar_file_path: str, extracted_folder_path: str):
    '''
    Creates a tensorflow dataset object from a tar file

    Args:
        tar_file_path (str): Path to a tar.gz file
        extracted_folder_path (str): Path for the extracted data to go
    '''

    with tarfile.open(tar_file_path, 'r:gz') as tar:
        tar.extractall(path=extracted_folder_path)


def pre_process_trimap(trimap: tf.Tensor, isCat: bool) -> tf.Tensor:
    '''
    Preprocesses a trimap tensor so all pixels which represent cats have a
    value of 2, all pixels which represent dogs have a value of 1, and all
    pixels which do not represent anything have a value of 0

    Args:
        trimap (tf.Tensor): Oxford-iiit pet dataset trimap
        isCat (bool): Boolean representing whether the image contains a cat or
            dog
    '''

    #clips all data to be either zero or one
    trimap = tf.where(tf.equal(trimap, 2), tf.zeros_like(trimap), trimap)
    trimap = tf.where(tf.equal(trimap, 3), tf.ones_like(trimap), trimap)


    #makes cat tensor to be 2 if it represents a cat
    if isCat:
        trimap *= 2

    return trimap


def preprocess_image(image_path: str, trimap_path: str) -> tuple:
    '''
    Preprocesses a data-label pair

    Args:
        image_path (str): Path to an image .jpg file
        trimap_path (str): Path to a trimap .png file
    '''


    #all cats in this dataset start with a capital letter
    image_path_prefix = "data/images/"
    image_first_letter = tf.strings.substr(image_path, pos=len(image_path_prefix), len=1)
    print(image_first_letter)
    isCat = tf.strings.regex_full_match(image_first_letter, '[A-Z]')

    print(isCat)

    # isCat = bool(isCat.numpy())    

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    trimap = tf.io.read_file(trimap_path)
    trimap = tf.image.decode_png(trimap, channels=1)

    trimap = pre_process_trimap(trimap, isCat)

    return image, trimap


def create_image_trimap_dataset(data_path: str) -> tf.data.Dataset:
    '''
    Creates a tensorflow dataset from a data directory

    Args:
        data_path (str): Path to a dataset
    Returns:
        tf.data.Dataset
            Tensorflow dataset mapping images to trimaps
    '''

    image_directory = data_path + "/images"
    trimap_directory = data_path + "/annotations/trimaps"

    image_filenames = os.listdir(image_directory)

    image_trimap_paths = [(os.path.join(image_directory, filename),
                           os.path.join(trimap_directory, filename.replace(".jpg", ".png")))
                           for filename in image_filenames]
    
    image_trimap_paths
    
    dataset = tf.data.Dataset.from_tensor_slices(image_trimap_paths)

    return dataset



# data_dir = Path('data').with_suffix('')

# images = list(data_dir.glob('annotations/trimaps/*'))
# print(len(images))
# im = PIL.Image.open(str(images[0]))

# im.show()

image_directory = "data/images"
trimap_directory = "data/annotations/trimaps"

image_filenames = os.listdir(image_directory)

image_paths = [os.path.join(image_directory, filename) for filename in image_filenames]

trimap_paths = [os.path.join(trimap_directory, filename.replace(".jpg", ".png"))
                for filename in image_filenames]

input_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
target_dataset = tf.data.Dataset.from_tensor_slices(trimap_paths)

dataset = tf.data.Dataset.zip((input_dataset, target_dataset))



dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# batches = (
#     dataset
#     .cache()
#     .shuffle(1000)
#     .batch(64)
#     .repeat()
#     .prefetch(buffer_size=tf.data.AUTOTUNE)
# )

first = dataset.take(1)
first = next(iter(first.as_numpy_iterator()))
print(first)

# plt.imshow(tf.keras.utils.array_to_img(first[0]))




# for images, masks in batches.take(2):
#     sample_image, sample_mask = images[0], masks[0]

#     plt.imshow(tf.keras.utils.array_to_img(sample_mask))
#     plt.axis('off')
#     plt.show()






# dataset = dataset.as_numpy_iterator()
# print(list(dataset)[0])
