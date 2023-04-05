import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from Dataframe import Dataframe
from Dataset import Dataset
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import params

dataframe = Dataframe()
df = dataframe.get_dataframe()


print(len(df))
df = df[: int(params.scale * len(df))]
print(len(df))

temp_median = df["Température"].median()

# remplacement des valeurs nulles par la mediane
df["Température"].fillna(temp_median, inplace=True)

train_len = int(params.train_prop * len(df))


df.isnull().sum()
df.fillna(df["Température_(°C)"].mean(), inplace=True)

dataset = Dataset(df, train_len, params.features)
dataset_train, dataset_test = dataset.get_dataset()

train_generator = TimeseriesGenerator(
    dataset_train,
    dataset_train,
    length=params.sequence_len,
    batch_size=params.batch_size,
)
test_generator = TimeseriesGenerator(
    dataset_test, dataset_test, length=params.sequence_len, batch_size=params.batch_size
)

# ---- About

x, y = train_generator[0]
print(f"Nombre de train batchs disponibles : ", len(train_generator))
print("batch x shape : ", x.shape)
print("batch y shape : ", y.shape)

x, y = train_generator[0]

save_dir = f"./best_model.h5"
bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_dir, verbose=0, save_best_only=True
)

model = keras.models.Sequential()
model.add(
    keras.layers.InputLayer(input_shape=(params.sequence_len, params.features_len))
)
model.add(keras.layers.LSTM(100, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(params.features_len))

model.summary()

model.compile(optimizer="adam", loss="mse", metrics=["mae"])


def plot_history(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


history = model.fit(
    train_generator,
    epochs=params.epochs,
    verbose=params.fit_verbosity,
    validation_data=test_generator,
    callbacks=[bestmodel_callback],
)
plot_history(history)
