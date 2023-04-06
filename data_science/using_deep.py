import random

import matplotlib
import os
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_science.Dataset import Dataset
from data_science.Dataframe import Dataframe
import data_science.params as params
import plotly.express as px


def predict(nb_predictions, path, position):

    loaded_model = tf.keras.models.load_model(f"data_science/best_model.h5")

    dataframe = Dataframe(path)
    df = dataframe.get_dataframe()
    train_len = int(params.train_prop * len(df))
    dataset = Dataset(df, train_len, params.features, params.features_w_date)
    mean = dataset.get_mean()
    std = dataset.get_std()

    dataset_train, dataset_test = dataset.get_dataset()
    dataset_train_copy, dataset_test_copy = dataset.get_dataset_copy()


    s = random.randint(0, len(dataset_test) - params.sequence_len - nb_predictions)

    sequence_pred = dataset_test[s : s + params.sequence_len].copy()
    sequence_true = dataset_test[s : s + params.sequence_len + nb_predictions].copy()

    sequence_pred_date = dataset_test_copy[s:s+params.sequence_len].copy()
    sequence_true_date = dataset_test_copy[s:s+params.sequence_len+nb_predictions].copy()

    # ---- Iterate on 4 predictions

    sequence_pred = list(sequence_pred)

    for i in range(nb_predictions):
        sequence = sequence_pred[-params.sequence_len :]
        print('len pred', len(np.array(sequence)))
        print('val pred', np.array([sequence]))
        pred = loaded_model.predict(np.array([sequence]))
        sequence_pred.append(pred[0])

    # ---- Extract the predictions

    pred = np.array(sequence_pred[-nb_predictions :])
    sequence_true, pred , sequence_pred_date, sequence_true_date = get_prediction(
    dataset_test, mean, std, loaded_model, dataset_test_copy, nb_predictions, params.sequence_len
    )
    print(pred)
    print(sequence_true)

    # Supposons que les données d'apprentissage et de test sont stockées dans des tableaux NumPy
    # X_train contient les données des 16 premiers jours (16 x T x N) où T est le nombre de temps et N est le nombre de variables
    # y_train contient les valeurs réelles de la température au 17ème jour (T x 1)
    # y_pred contient les valeurs prédites de la température au 17ème jour (1 x 1)
    if position== 0:
        deuxieme_colonne = [ligne[position]-273.15 for ligne in sequence_true]
        pred_co = [ligne[position] -273.15 for ligne in pred]
    else:
        deuxieme_colonne = [ligne[position] for ligne in sequence_true]
        pred_co = [ligne[position] for ligne in pred]


    # print(deuxieme_colonne)
    # print(deuxieme_colonne[: -nb_predictions])

    # print(pred_co)

    # On récupère la valeur réelle de la température au 17ème jour
    y_true = deuxieme_colonne[-nb_predictions :]

    print(y_true)
    # On trace la courbe de la température pour les 16 premiers jours
    # plt.plot(deuxieme_colonne[: -nb_predictions])

    # # On trace la valeur réelle de la température au 17ème jour en rouge
    # for i in range(nb_predictions):
    #     print(y_true[i])
    #     print(i + 1)

    #     plt.plot((len(deuxieme_colonne) - nb_predictions) + (i + 1), y_true[i], "ro")

    # # On trace la valeur prédite de la température au 17ème jour en vert
    # for i in range(nb_predictions):
    #     plt.plot((len(deuxieme_colonne) - nb_predictions) + (i + 1), pred_co[i], "go")

    # # On ajoute des labels pour l'axe x et l'axe y
    # plt.xlabel("Temps")
    # plt.ylabel("Température")

    # # On affiche le graphique
    # plt.show()

    # fig = px.ecdf(df, x="total_bill", color="sex")
    # fig.show()

    return deuxieme_colonne, pred_co, sequence_pred_date, sequence_true_date


def denormalize(mean, std, seq):
    nseq = seq.copy()
    for i, s in enumerate(nseq):
        s = s * std + mean
        nseq[i] = s
    return nseq


def get_prediction(dataset,mean, std, model, dataset_test_copy, iterations=4, sequence_len=16, ):

    # ---- Initial sequence

    s = random.randint(0, len(dataset) - sequence_len - iterations)

    sequence_pred = dataset[s : s + sequence_len].copy()
    sequence_true = dataset[s : s + sequence_len + iterations].copy()


    sequence_pred_date = dataset_test_copy[s+sequence_len:s+sequence_len+iterations].copy()
    sequence_true_date = dataset_test_copy[s:s+sequence_len+iterations].copy()

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

    return sequence_true, pred, sequence_pred_date, sequence_true_date



