import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
from IPython.display import clear_output


base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int) -> tf.keras.Model:
    '''
    Creates a U-Net model with a pre-trained down-stack. Input is a 128x128 three
    channel image, and the output is a 128x128 image with a variable amount of
    output channels

    Args:
        output_channels (int): number of output channels (labels) for the
        128x128 output
    Returns:
        tf.keras.Model
            U-Net model defined by the parameters above, with four skip
            connections and a final output layer which is an up-convulution
            to result in a 128x128 output
    '''
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def display(display_list: list):
    '''
    Displays from a list of images

    Args:
        display_list (list): List of tensor images
    '''
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']


    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        # if i == 2:
        #     if max(display_list[i] == 2):
        #         print("Predicted class: Cat")
        #     else:
        #         print("Predicted class: Dog")
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    '''
    Creates a mask from a predicted mask

    Args:
        pred_mask (tf.Tensor): Predicted 3-channel output mask
    Returns:
        tf.Tensor
            the predicted tensor of the model
    '''
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model: tf.keras.Model, sample_image: tf.Tensor, sample_mask: tf.Tensor):
    '''
    Shows the predictions of a model

    Args:
        model (tf.keras.model): model
        sample_image (tf.Tensor): a sampled image
        sample_mask (tf.Tensor): a sampled mask
    '''

    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])