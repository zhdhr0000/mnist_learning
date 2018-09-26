###学习keras api的使用方法, 1 mnist数据集
import keras
mnist = keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28,28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28,28)
    input_shape = (1, 28,28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28,28, 1)
    x_test = x_test.reshape(x_test.shape[0],28,28, 1)
    input_shape = (28,28, 1)

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

print('x_train shape:',x_train.shape)
print('y_train shape:',y_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu", input_shape=input_shape))
model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10,activation="softmax"))

  # 原本的普通的全连接网络
  # tf.keras.layers.Flatten(),
  # tf.keras.layers.Dense(512, activation=tf.nn.relu),
  # tf.keras.layers.Dropout(0.2),
  # tf.keras.layers.Dense(512, activation=tf.nn.relu),
  # tf.keras.layers.Dropout(0.2),
  # tf.keras.layers.Dense(10, activation=tf.nn.softmax)


# model.load_weights('./my_model_weights')
# .load('my_model')
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, verbose=1,epochs=160,validation_split=1/12)
score = model.evaluate(x_test, y_test,verbose=0)

print(score[0])
print(score[1])

model.save('./mnist_model.h5')

### model序列化为JSON和yaml的方法
# Serialize a model to JSON format
# json_string = model.to_json()

# Recreate the model (freshly initialized)
# fresh_model = keras.models.from_json(json_string)

# Serializes a model to YAML format
# yaml_string = model.to_yaml()

# Recreate the model
# fresh_model = keras.models.from_yaml(yaml_string)
