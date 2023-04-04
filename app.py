from flask import Flask
from Prediction import Prediction
from flask_restful import  Api 

app = Flask("Konoha_Team7")

api = Api(app)

api.add_resource(Prediction, '/prediction/<prediction_id>')

if __name__ == '__main__':
    app.run(debug=True, port=9000)