import pandas as pd
import data_science.params as params
from flask import Flask, request, render_template, redirect, url_for
import os

class Dataframe:

    # default constructor
    def __init__(self, path_csv):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(dir_path, 'assets')
        current_file_path = os.path.abspath(__file__)
        # print('current path',current_file_path)
        # Obtenir le chemin absolu vers le répertoire parent du script actuel
        project_dir_path = os.path.dirname(current_file_path)
        # print('project_dir_path path',project_dir_path)

        self.df = pd.read_csv(os.path.join(assets_dir, path_csv))
        # Création de la dataframe avec une colonne 'date'
        self.df["Date_Heure"] = pd.to_datetime(
            self.df["Date_Heure"]
        )
        self.df.sort_values(['Date_Heure'],  inplace=True)
        # Encodage one-hot de la colonne 'date'
        self.df[["Spring", "Summer", "Autumn", "Winter"]] = (
            self.df["Date_Heure"].apply(self.one_hot_season).apply(pd.Series)
        )
        # Encodage one-hot de la colonne 'date'
        self.df[["week", "weekend"]] = (
            self.df["Date_Heure"].apply(self.one_hot_weekday).apply(pd.Series)
        )

        temp_median = self.df["Température"].median()

        # remplacement des valeurs nulles par la mediane
        self.df["Température"].fillna(temp_median, inplace=True)
        self.df.columns = self.df.columns.str.replace(" ", "_")
        self.df = self.df[: int(params.scale * len(self.df))]

    def one_hot_season(self, date):
        month = date.month
        if month in (3, 4, 5):
            return [1, 0, 0, 0]  # Spring
        elif month in (6, 7, 8):
            return [0, 1, 0, 0]  # Summer
        elif month in (9, 10, 11):
            return [0, 0, 1, 0]  # Autumn
        else:
            return [0, 0, 0, 1]  # Winter

    def one_hot_weekday(self, date):
        weekday = date.weekday()
        if weekday in (0, 1, 2, 3, 4):
            return [1, 0]  # week
        else:
            return [0, 1]  # weeking

    def get_dataframe(self):
        return self.df
