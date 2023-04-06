from flask import Flask, request, render_template, redirect, url_for, jsonify
from Prediction import Prediction
from flask_restful import Api
import os
import data_science.using_deep as predict
import data_science.params as params
import json
import plotly

app = Flask("Konoha_Team7")

api = Api(app)

UPLOAD_FOLDER = 'assets/'

@app.route("/prediction/<prediction_id>")
def prediction():

    return Prediction.get()


@app.route("/", methods=['GET', 'POST'])
def home():
    title = "Konoha_Team7"
    return render_template("index.html.j2", title=title)

@app.route("/predict", methods=['POST'])
def predict_new():
    title = "Konoha_Team7"
    if request.method == 'POST':
        print('in post')
        # Vérifie si le fichier est bien présent dans la requête
        number = int(request.form['number'])
        print('nombre', number)
        print('ttoot')
        # if 'file' not in request.files:
        #     print('not file in')
        #     return redirect(request.url)
        # file = request.files['file']
        # # Vérifie si le nom de fichier est vide
        # if file.filename == '':
        #     print('empty')
        #     return redirect(request.url)
        # # Enregistre le fichier dans le dossier assets
        # if file:
        #     filename = file.filename
        #     print('filename', filename)
        #     file.save(os.path.join(UPLOAD_FOLDER, filename))
        #     # return redirect(url_for('display_file', filename=filename))
        current_file_path = os.path.abspath(__file__)
        print('current path',current_file_path)
        # Obtenir le chemin absolu vers le répertoire parent du script actuel
        project_dir_path = os.path.dirname(current_file_path)
        assets_dir = os.path.join(project_dir_path, 'assets')
        path_final = os.path.join(assets_dir, params.path)
        # print('project_dir_path path',project_dir_path)
        # print('assets_dir path',assets_dir)
        # print('path_final path',path_final)
        reel_conso_value, predicted_conso_value = predict.predict(number, path_final,1)
        reel_temp_value, predicted_temp_value = predict.predict(number, path_final, 0)
        # print("real", a)
        # print("predict", b)

        data = {
            'consomation':make_graph(predicted_conso_value,reel_conso_value, 'Consomation'),
            'temperature': make_graph(predicted_temp_value,reel_temp_value, 'Température (°C)')
        }
        
    return jsonify(data)


def make_graph(valeurs_predites,valeurs_reelles, graph_title):
    import plotly.graph_objects as go

    # données réelles et prédictions
    # valeurs_reelles = [25, 23, 24, 22, 27, 26, 28, 29, 31, 30]
    # valeurs_predites = [25, 24, 26, 27]
    print(valeurs_predites)
    print(valeurs_reelles)

    # indices correspondants
    indices_reels = range(len(valeurs_reelles))
    indices_predits = range(len(valeurs_reelles) - len(valeurs_predites), len(valeurs_reelles))

    # tracer la courbe de points pour les valeurs réelles
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(indices_reels), y=valeurs_reelles,
                            mode='markers+lines',
                            name='Valeurs réelles',
                            line=dict(color='blue', width=2)))

    # tracer la courbe de points pour les valeurs prédites
    fig.add_trace(go.Scatter(x=list(indices_predits), y=valeurs_predites,
                            mode='markers+lines',
                            name='Valeurs prédites',
                            line=dict(color='red', width=2, dash='dot')))

    # personnalisation de l'axe x
    fig.update_xaxes(title='Temps')

    # personnalisation de l'axe y
    fig.update_yaxes(title=graph_title)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # afficher la figure
    # fig.show()
    return graphJSON

if __name__ == '__main__':
    app.run(debug=True, port=9000)
