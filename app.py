from flask import Flask,jsonify
from product_review import product_review_analysis, competition_analysis, sales_potential
import pyrebase
import time
from flask_cors import CORS,cross_origin
# to connect to firebase
config = {
    'apiKey': "AIzaSyBuahgMi-FMIl-3_WSd2Y8CoClwdxJFurU",
    'authDomain': "reviewisaproduct.firebaseapp.com",
    'databaseURL': "https://reviewisaproduct-default-rtdb.firebaseio.com",
    'projectId': "reviewisaproduct",
    'storageBucket': "reviewisaproduct.appspot.com",
    'serviceAccount':"reviewisaproduct-firebase-adminsdk-6j199-58b96a44fa.json"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
auth = firebase.auth()
user = auth.sign_in_with_email_and_password("reviewisaproduct@gmail.com","SXC@2021")
db = firebase.database()
app = Flask(__name__)

# route which follows to run the desired functions
@app.route('/')
def hello():
    return ("Hi")

@app.route('/product_review/<asin>')
@cross_origin()
def product_review(asin="B000AST3AK"):
    # B001B35APA
    # B01G91Y4VE
    # B00570QQ5G
    # B004DNWVPC
    # B01CTNA1VI
    # B00KJ07SEM
    # 'B000AST3AK', 'B004UB1O9Q', 'B0014CN8Y8', 'B00KJ07SEM', 'B0045LLC7K',
    # 'B001B35APA', 'B00INXG9MY', 'B00E37TQV0', 'B004INUWX0', 'B01H1R0K68'
    ts = str(time.time())
    product_rvw = product_review_analysis(asin, storage, user, ts)
    competition = competition_analysis(asin, storage, user, ts)
    sales_p = sales_potential(asin, product_rvw, competition, storage, user, ts)
    # forming a json object to pass as response to flask get request
    analysis = {}
    for outputs in [product_rvw, competition, sales_p]:
        for key in outputs.keys():
            analysis[key] = outputs[key]
    # pushing the json object to firebase realtime database
    key = db.push(analysis)
    analysis['key'] = str(key['name'])
    # returning the json object
    return jsonify(analysis)

# running the flask app
if __name__ == '__main__':
    app.run()

