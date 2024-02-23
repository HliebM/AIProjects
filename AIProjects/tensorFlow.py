import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10)
# model.save("model")

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print('\nTest accuracy:', test_acc)



imported_model = tf.keras.models.load_model("model")
probability_model = tf.keras.Sequential([imported_model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print(predictions[0][0]*100, "%", " - 0\n")
print(predictions[0][1]*100, "%", " - 1\n")
print(predictions[0][2]*100, "%", " - 2\n")
print(predictions[0][3]*100, "%", " - 3\n")
print(predictions[0][4]*100, "%", " - 4\n")
print(predictions[0][5]*100, "%", " - 5\n")
print(predictions[0][6]*100, "%", " - 6\n")
print(predictions[0][7]*100, "%", " - 7\n")
print(predictions[0][8]*100,  "%", " - 8\n")
print(predictions[0][9]*100, "%", " - 9\n")
print(test_labels[0], "- correct answer")