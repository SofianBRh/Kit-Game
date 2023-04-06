from flask import Flask, render_template, redirect, url_for
from Prediction import Prediction
from flask_restful import Api 

app = Flask("Konoha_Team7")

api = Api(app)

@app.route("/prediction/<prediction_id>")
def prediction():
    
    return Prediction.get()

@app.route("/")
def home():
    title = "Konoha_Team7"
    return render_template("index.html.j2", title=title)


if __name__ == '__main__':
    app.run(debug=True, port=9000)