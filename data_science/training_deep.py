import matplotlib
import os
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from Dataframe import Dataframe
from Dataset import Dataset
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import params
from tensorflow.keras.callbacks import EarlyStopping


# tf.config.list_physical_devices("GPU")
# sys_details = tf.sysconfig.get_build_info()
# cuda = sys_details["cuda_version"]
# cudnn = sys_details["cudnn_version"]
# print(cuda, cudnn)

current_file_path = os.path.abspath(__file__)
print('current path',current_file_path)
# Obtenir le chemin absolu vers le répertoire parent du script actuel
project_dir_path = os.path.dirname(current_file_path)
main_dir = os.path.dirname(project_dir_path)
path_final = os.path.join(main_dir, 'assets')
path = os.path.join(path_final, params.path)
print('project_dir_path path',project_dir_path)
print('assets_dir path',main_dir)
print('path_final path',path_final)
dataframe = Dataframe(path)
df = dataframe.get_dataframe()

train_len = int(params.train_prop * len(df))


dataset = Dataset(df, train_len, params.features, params.features_w_date)
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
    filepath=save_dir, verbose=0, save_best_only=True, monitor='val_loss', patience=5, mode='min', restore_best_weights=True
)

# Callback EarlyStopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)


model = keras.models.Sequential()
model.add(
    keras.layers.InputLayer(input_shape=(params.sequence_len, params.features_len))
)
for i in range(1, 3):
    model.add( keras.layers.LSTM(100, activation='relu', return_sequences=True) )
    model.add( keras.layers.Dropout(0.2) )

model.add( keras.layers.LSTM(100, activation='relu') )
model.add( keras.layers.Dropout(0.2) )

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
