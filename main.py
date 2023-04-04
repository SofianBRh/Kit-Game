from flask import Flask
from flask_restful import Resource, Api

app = Flask("Konoha_Team7")
api = Api(app)

prediction = {
    'pred1':'ddsds',
    'pred2':'bdff'
}

class Prediction(Resource):

    def get(self,prediction_id):
        if prediction_id == '':
            return None
        return prediction[prediction_id]

api.add_resource(Prediction , '/prediction/<prediction_id>')

if __name__ == '__main__':
    app.run(debug=True, port=9000)