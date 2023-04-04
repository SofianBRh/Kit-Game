from Prediction import Prediction
from app import api

api.add_resource(Prediction, '/prediction/<prediction_id>')
