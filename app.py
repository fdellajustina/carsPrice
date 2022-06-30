from flask import Flask, request
from flask import render_template
from predict.car_price import PredictCarPrice

app = Flask(__name__)

@app.route("/")
def index():
    result = None
    return render_template("index.html", result = result)

@app.route("/estimate", methods = ["POST"])
def estimate():
    values = request.form.getlist('new_car')
    
    car = PredictCarPrice(values)
    value_to_predict = car.prepare()
    result = car.predict(value_to_predict)
    result = "%.2f" % result
    return render_template('index.html', result = result)

if __name__ == "__main__":
    app.run(host = 'localhost', port = 3000, debug = True)
