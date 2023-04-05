import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# librairies de data viz
import matplotlib.pyplot as plt
import seaborn as sns
# librairies des modèle linéaire
from sklearn import linear_model
# librairie des modèle d'ensemble
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
# librairies des métriques
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
# librairies pour le k-fold cross validation
from sklearn.model_selection import KFold, cross_val_score
import datetime as dt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

df = pd.read_csv("./assets/donnees.csv")  # READ INPUT CSV FILE
# print(df.head()) # PRINT A FEW LINES
# print('Nbr de lignes et nbr de colonnes : ',df.shape) # PRINT LINES AND COLUMNS COUNTS
# print(df.dtypes) # PRINT COLUMNS TYPES

percent_missing = df.isnull().sum() * 100 / len(df)
# PRINT MISSING DATA PERCENTAGE
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
# print(missing_value_df)

df.columns = df.columns.str.replace(' ', '_')  # REPLACE SPACES WITH _
# print(df.head())

# CREATE HEATMAP WITH SPECIFIED COLUMS
corr_one = df[['index', 'Pression_au_niveau_mer', 'Variation_de_pression_en_3_heures', 'Type_de_tendance_barométrique',
               'Direction_du_vent_moyen_10_mn', 'Température', 'Point_de_rosée', 'consommation', 'Humidité']].corr()
index_one = corr_one.index
heatmap_one = sns.heatmap(df[index_one].corr(), annot=True)
# SAVE HEATMAP TO PNG FILE
plt.savefig("heatmap_one.png", orientation='landscape',
            format="png", bbox_inches="tight")

# CREATE HEATMAP WITH SPECIFIED COLUMS
corr_two = df[['Visibilité_horizontale', 'Temps_présent', 'Pression_station', 'Rafales_sur_une_période', 'Periode_de_mesure_de_la_rafale',
               'Précipitations_dans_la_dernière_heure', 'Précipitations_dans_les_3_dernières_heures', 'Température_(°C)', 'consommation', 'datehour', 'datemonth']].corr()
index_two = corr_two.index
heatmap_two = sns.heatmap(df[index_two].corr(), annot=True)
# SAVE HEATMAP TO PNG FILE
plt.savefig("heatmap_two.png", orientation='landscape',
            format="png", bbox_inches="tight")

# COUNT MISSING DATA
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})

# REFORMAT DATES
df['Date_Heure'] = pd.to_datetime(df['Date_Heure'], format=('%Y-%m-%d %H:%M:%S'))

# EXTRACT DATE INFOS INTO SPECIFIC FIELDS
df['year'] = df['Date_Heure'].dt.year
df['month'] = df['Date_Heure'].dt.month
df['day'] = df['Date_Heure'].dt.day

# print(df['year'])
# print(df['month'])
# print(df['day'])
# REPLACE NULL DATAS WITH SPECIFIC
df.replace(0, np.nan, inplace=True)

df.fillna(df.mean(), inplace=True)
df.isnull().sum()


# CREATE TEST DATASET
y = df[['consommation']]
X = df.drop(['consommation',"Date_Heure", 'Pression_au_niveau_mer', 'index', 'Type_de_tendance_barométrique', 'Direction_du_vent_moyen_10_mn', 'Humidité',
            'Visibilité_horizontale', 'Temps_présent', 'Rafales_sur_une_période', 'Periode_de_mesure_de_la_rafale', 'Précipitations_dans_la_dernière_heure'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# CREATE XGBRegressor MODEL
model = XGBRegressor(n_estimators=150, max_depth=6, eta=0.05, subsample=0.8, colsample_bytree=0.8,verbosity=1)
model.fit(X_train, Y_train)
score = model.score(X_train, Y_train)
print(f"********* score: {score}  ***********")
scores = cross_val_score(model, X, y.values.ravel(), cv=2)
print("scores",scores)
print("moyenne :%0.03f , deviation: :%0.03f" % (scores.mean(), scores.std()))


kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(model, X_train, Y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

ypred = model.predict(X_test)
mse = mean_squared_error(Y_test, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))

plt.close('all')

x_ax = range(len(Y_test))

plt.scatter(x_ax, Y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()