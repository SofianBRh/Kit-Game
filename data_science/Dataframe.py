import pandas as pd


class Dataframe:

    # default constructor
    def __init__(self):
        self.df = pd.read_csv("/home/kyabe/cours/kitgames/Kit-Game/assets/donnees.csv")
        # Création de la dataframe avec une colonne 'date'
        self.df["Date_Heure"] = pd.to_datetime(
            self.df["Date_Heure"], format=("%Y-%m-%dT%H:%M:%S.%f")
        )
        # Encodage one-hot de la colonne 'date'
        self.df[["Spring", "Summer", "Autumn", "Winter"]] = (
            self.df["Date_Heure"].apply(self.one_hot_season).apply(pd.Series)
        )
        # Encodage one-hot de la colonne 'date'
        self.df[["week", "weekend"]] = (
            self.df["Date_Heure"].apply(self.one_hot_weekday).apply(pd.Series)
        )
        self.df.columns = self.df.columns.str.replace(" ", "_")

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
