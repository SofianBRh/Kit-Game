features = [
    "Température",
    "consommation",
    "weekend",
    "week",
    "Winter",
    "Autumn",
    "Summer",
    "Spring",
]

features_w_date   = ['Température', 'consommation', 'weekend', 'Date_Heure', 'week', 'Winter', 'Autumn', 'Summer', 'Spring']
features_len = len(features)

scale = 0.9  # Percentage of dataset to be used (1=all)
train_prop = 0.8  # Percentage for train (the rest being for the test)
sequence_len = 16
batch_size = 32
epochs = 150
fit_verbosity = 1
iterations = 10
path = "donnees.csv"
lstm = 80
dropout = 0.2
