import random

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Dataset import Dataset
from Dataframe import Dataframe
import params

loaded_model = tf.keras.models.load_model(f"./best_model.h5")

dataframe = Dataframe()
df = dataframe.get_dataframe()
train_len = int(params.train_prop * len(df))
dataset = Dataset(df, train_len, params.features)
mean = dataset.get_mean()
std = dataset.get_std()

dataset_train, dataset_test = dataset.get_dataset()


s = random.randint(0, len(dataset_test) - params.sequence_len - params.iterations)

sequence_pred = dataset_test[s : s + params.sequence_len].copy()
sequence_true = dataset_test[s : s + params.sequence_len + params.iterations].copy()

# ---- Iterate on 4 predictions

sequence_pred = list(sequence_pred)

for i in range(params.iterations):
    sequence = sequence_pred[-params.sequence_len :]
    pred = loaded_model.predict(np.array([sequence]))
    sequence_pred.append(pred[0])

# ---- Extract the predictions

pred = np.array(sequence_pred[-params.iterations :])


def denormalize(mean, std, seq):
    nseq = seq.copy()
    for i, s in enumerate(nseq):
        s = s * std + mean
        nseq[i] = s
    return nseq


def get_prediction(dataset, model, iterations=4, sequence_len=16):

    # ---- Initial sequence

    s = random.randint(0, len(dataset) - sequence_len - iterations)

    sequence_pred = dataset[s : s + sequence_len].copy()
    sequence_true = dataset[s : s + sequence_len + iterations].copy()

    # ---- Iterate

    sequence_pred = list(sequence_pred)

    for i in range(iterations):
        sequence = sequence_pred[-sequence_len:]
        pred = model.predict(np.array([sequence]))
        sequence_pred.append(pred[0])

    # ---- Extract the predictions

    pred = np.array(sequence_pred[-iterations:])

    # ---- De-normalization

    sequence_true = denormalize(mean, std, sequence_true)
    pred = denormalize(mean, std, pred)

    return sequence_true, pred


sequence_true, pred = get_prediction(dataset_test, loaded_model, iterations=4)
print(pred)
print(sequence_true)

# Supposons que les données d'apprentissage et de test sont stockées dans des tableaux NumPy
# X_train contient les données des 16 premiers jours (16 x T x N) où T est le nombre de temps et N est le nombre de variables
# y_train contient les valeurs réelles de la température au 17ème jour (T x 1)
# y_pred contient les valeurs prédites de la température au 17ème jour (1 x 1)
deuxieme_colonne = [ligne[1] for ligne in sequence_true]
pred_co = [ligne[1] for ligne in pred]


print(deuxieme_colonne)
print(deuxieme_colonne[: -params.iterations])

print(pred_co)

# On récupère la valeur réelle de la température au 17ème jour
y_true = deuxieme_colonne[-params.iterations :]

print(y_true)
# On trace la courbe de la température pour les 16 premiers jours
plt.plot(deuxieme_colonne[: -params.iterations])

# On trace la valeur réelle de la température au 17ème jour en rouge
for i in range(params.iterations):
    print(y_true[i])
    print(i + 1)

    plt.plot((len(deuxieme_colonne) - params.iterations) + (i + 1), y_true[i], "ro")

# On trace la valeur prédite de la température au 17ème jour en vert
for i in range(params.iterations):
    plt.plot((len(deuxieme_colonne) - params.iterations) + (i + 1), pred_co[i], "go")

# On ajoute des labels pour l'axe x et l'axe y
plt.xlabel("Temps")
plt.ylabel("Température")

# On affiche le graphique
plt.show()
