import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

# build model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),  # sigmoid layer
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# create dataset

import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

# fit
model.fit(data, labels, epochs=10, batch_size=32)  # 一个周期（epoch）里1000个样本（全部data），一次跑32个样本

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

# validation
model.fit(data, labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30)  # 一个step是32个样本，30*32=960

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()
model.fit(dataset, epochs=10, steps_per_epoch=30, validation_data=val_dataset, validation_steps=3)

# evaluate
model.evaluate(data, labels, batch_size=32)
model.evaluate(dataset, steps=30)

# predict
result = model.predict(data, batch_size=32)
print(result.shape)

# functional api
inputs = tf.keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)


# model class
class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

model = MyModel(num_classes=10)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, batch_size=32, epochs=5)
