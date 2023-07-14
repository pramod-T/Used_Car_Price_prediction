from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = open('car.pkl','rb')
regressor = pickle.load(model)
model.close()

@app.route("/")
def home():
    return render_template('car_web.html')

@app.route('/predict', methods=["POST"])
def predict():
    name = request.form.get("Brand")
    loc = request.form.get("Location")
    fuel = request.form.get("Fuel")
    trans = request.form.get("Transmission")     
    Year = int(request.form.get("Year"))
    Kms = float(request.form.get("Kms"))
    Own = request.form.get("Owner")
    Mileage = float(request.form.get("Mileage"))
    Engine = float(request.form.get("Engine"))
    Power = float(request.form.get("Power"))
    Seat = request.form.get("Seats")
    car_age=2021-Year
#PREDICTION
    Price = regressor.predict([[name,loc,Year,Kms,fuel,trans,Own,Mileage,Engine,Power,Seat,car_age]])

    output=round(Price[0],2)

    return render_template('result.html',prediction_text="Your car's price should be Rs. {} lakhs. This price may change depending on the condition of the car.".format(output))


if __name__ == "__main__":
    app.run(debug=False)