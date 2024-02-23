import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_height = 28
img_width = 28
batch_size = 32

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    './img/cats/',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    validation_split=0.1,
    seed=123,
    subset='training'
)
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    './img/cats/',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    validation_split=0.1,
    seed=123,
    subset='validation'
)
ds_validation_new = tf.keras.preprocessing.image_dataset_from_directory(
    './img/other_cats/',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
)
model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same'),
    tf.keras.layers.Conv2D(32, 3, padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10),
])

def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y

ds_train = ds_train.map(augment)

for epochs in range(10):
    for x, y in ds_train:
        pass


model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
              loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
              metrics=['accuracy'])

# model.fit(ds_train, epochs=500, verbose=2)
# model.save("detect_noise")

imported_model = tf.keras.models.load_model("detect_noise")
probability_model = tf.keras.Sequential([imported_model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(ds_validation_new)



for allPredictions in predictions:
    print(list(allPredictions))
    i = 0
    for prediction in allPredictions:
        print(i, '-', f'{prediction*100}%')
        i += 1
    print("AI: my answer is", np.argmax(allPredictions))

