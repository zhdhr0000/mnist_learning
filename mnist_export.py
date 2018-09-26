import keras

if keras.backend.image_data_format() == 'channels_first':
    input_shape = (1, 28,28)
else:
    input_shape = (28,28,1)
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu", input_shape=input_shape))
model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10,activation="softmax"))

model.load_weights('./mnist_model.h5')

# model.save('./mnist_model.h5')

keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')


# tflite_convert \
#   --output_file=/Users/zhangyh/Desktop/mnist_learning/tmp/mnist_model.tflite \
#   --keras_model_file=/Users/zhangyh/Desktop/mnist_learning/mnist_model.h5