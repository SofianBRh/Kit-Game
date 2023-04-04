from flask_restful import Resource


class Prediction(Resource):

    prediction = {
        'pred1': 'ddsds',
        'pred2': 'bdff'
    }

    def get(self, prediction_id):
        if prediction_id == '':
            return None
        return self.prediction[prediction_id]
