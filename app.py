from flask import Flask
from flask_restful import  Api 
from routes import 

app = Flask("Konoha_Team7")
api = Api(app)

if __name__ == '__main__':
    app.run(debug=True, port=9000)