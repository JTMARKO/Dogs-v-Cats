import tensorflow as tf
import tarfile
import numpy as np
import os
from pathlib import Path

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


def pre_process_trimap(trimap: tf.Tensor, isCat: tf.bool) -> tf.Tensor:
    '''
    Preprocesses a trimap tensor so all pixels which represent cats have a
    value of 2, all pixels which represent dogs have a value of 1, and all
    pixels which do not represent anything have a value of 0

    Args:
        trimap (tf.Tensor): Oxford-iiit pet dataset trimap
        isCat (bool): Boolean representing whether the image contains a cat or
        dog
    Returns:
        tf.Tensor
            Tensor mapping representing the edited pixels in the png.
    '''

    # clips all data to be either zero or one; data has the mappings
    # 1:foreground, 2:background, 3:edges. this removes the background so only
    # the mask remains
    trimap %= 2

    # makes cat tensor to be 2 if it represents a cat
    if isCat:
        trimap *= 2

    return trimap


def preprocess_image(image_path: tf.string, trimap_path: tf.string) -> tuple:
    '''
    Preprocesses a data-label pair

    Args:
        image_path (str): Path to an image .jpg file
        trimap_path (str): Path to a trimap .png file
    Returns:
        tuple
            Tuple of tensor objects representing i/o of a dataset in the shape
            (128x128)
    '''


    # all cats in this dataset have a filename which starts with a capital letter
    first_letter_position = len("Data/images/")

    image_first_letter = tf.strings.substr(image_path, pos=first_letter_position, len=1)
    isCat = tf.strings.regex_full_match(image_first_letter, '[A-Z]')


    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image must be in range 0-1
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (128, 128),)

    trimap = tf.io.read_file(trimap_path)
    trimap = tf.image.decode_png(trimap, channels=1)
    trimap = tf.image.resize(trimap,
                             (128, 128),
                             method = tf.image.ResizeMethod.NEAREST_NEIGHBOR, )

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

    image_paths = [os.path.join(image_directory, filename)
                   for filename in image_filenames ]

    trimap_paths = [os.path.join(trimap_directory, filename.replace(".jpg", ".png"))
                    for filename in image_filenames ]

    input_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    target_dataset = tf.data.Dataset.from_tensor_slices(trimap_paths)

    dataset = tf.data.Dataset.zip((input_dataset, target_dataset))

    return dataset


def split_dataset(dataset: tf.data.Dataset, training: float, testing: float) -> tuple:
    '''
    Splits a dataset three ways between training, testing and validation

    Args:
        dataset (tf.data.Dataset): the dataset to be split
        training (tf.data.Dataset): fraction of dataset to train
        testing (tf.data.Dataset): fraction of dataset to test
    Returns:
        tuple
            Tuple of (train_dataset, val_dataset, test_dataset)
    '''

    dataset = dataset.shuffle()

    validation = 1 - training - testing

    num_samples = tf.data.experimental.cardinality(dataset).numpy()

    train_samples = int(training * num_samples)
    val_samples = int(validation * num_samples)


    train_dataset = dataset.take(train_samples)
    val_dataset = dataset.skip(train_samples).take(val_samples)
    test_dataset = dataset.skip(train_samples + val_samples)

    return train_dataset, val_dataset, test_dataset





def augment_dataset(input_tensor: tf.Tensor, target_tensor: tf.Tensor) -> tuple:
    '''
    Augments an input/output pair with random rotations, zooms, and contrast
    changes

    Args:
        input_tensor (tf.Tensor): An input tensor
        output_tensor (tf.Tensor): An output tensor
    Returns:
        tuple
            Tuple of tensor objects representing i/o of a dataset
    '''
    pass







if __name__ == "__main__":

    dataset = create_image_trimap_dataset("Data")


    dataset = dataset.map(preprocess_image)


    batches = (
        dataset
        .cache()
        .shuffle(1000)
        .batch(64)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    dataset = dataset.shuffle(500)

    member_1 = dataset.take(1)

    member_233 = dataset.skip(232).take(1)


    for val in member_1:
        plt.imshow(val[0].numpy())
        plt.show()
        plt.imshow(val[1].numpy())
        plt.show()

    for val in member_233:
        plt.imshow(val[0].numpy())
        plt.show()
        plt.imshow(val[1].numpy())
        plt.show()

