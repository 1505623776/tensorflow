import tensorflow as tf

fashion = tf.keras.datasets.fashion_mnist
(train_data, train_target), (test_data, test_target) = fashion.load_data()

train_data, test_data = train_data/255.0, test_data/255.0

#model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)
model.fit(
    train_data, train_target, batch_size=128, epochs=10, validation_data=(test_data, test_target), validation_freq=1

)
model.summary()
