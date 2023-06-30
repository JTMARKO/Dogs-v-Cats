import tensorflow as tf

from augment_dataset import create_split_dataset
from build_model import unet_model, show_predictions



BATCH_SIZE = 64
BUFFER_SIZE = 1000

EPOCHS = 50
VAL_SUBSPLITS = 5
VALIDATION_STEPS = 5

OUTPUT_CLASSES = 3



if __name__ == "__main__":
    (train, val, test) = create_split_dataset("Data", .8, .1)

    TRAIN_LENGTH = tf.data.experimental.cardinality(train).numpy()
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    train_batches = (
        train
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    val_batches = val.batch(BATCH_SIZE)

    for images, masks in val_batches.take(2):
        # print(images)
        sample_image, sample_mask = images[0], masks[0]
        # display([sample_image, sample_mask])


    
    # model = unet_model(output_channels=OUTPUT_CLASSES)
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
            
    # # show_predictions(model, sample_image, sample_mask)

    # model_history = model.fit(train_batches, epochs=EPOCHS,
    #                           steps_per_epoch=STEPS_PER_EPOCH,
    #                           validation_steps=VALIDATION_STEPS,
    #                           validation_data=val_batches,)
    

    model = tf.keras.models.load_model("Models")

    for image, mask in test.skip(20).take(20):
        show_predictions(model, image, mask)


    