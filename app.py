from flask import Flask, render_template, request, redirect, session, jsonify
from pymongo import MongoClient
import pickle
from datetime import datetime

app = Flask(__name__)
app.secret_key = "smartcropsecret"

# Load ML model
model = pickle.load(open("model.pkl","rb"))
encoder = pickle.load(open("encoder.pkl","rb"))

# MongoDB Atlas Connection
client = MongoClient(
"mongodb+srv://24p31f0011_db_user:Saki1234%40%24%25@cluster0.whyc5tj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

db = client["smart_crop_db"]

users_collection = db["users"]
predictions_collection = db["predictions"]


# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("home.html")


# ---------------- SIGNUP ----------------

@app.route("/signup", methods=["GET","POST"])
def signup():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        users_collection.insert_one({
            "name":name,
            "email":email,
            "password":password,
            "role":"user"
        })

        return redirect("/login")

    return render_template("signup.html")


# ---------------- LOGIN ----------------

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = users_collection.find_one({
            "email":email,
            "password":password
        })

        if user:

            session["user_id"] = str(user["_id"])
            return redirect("/dashboard")

    return render_template("login.html")


# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():

    users = users_collection.count_documents({})

    predictions = predictions_collection.count_documents({})

    accuracy = 97

    # Latest soil moisture
    latest = predictions_collection.find().sort("_id",-1).limit(1)

    soil = 0

    for x in latest:
        soil = x["soil_moisture"]

    return render_template(
        "dashboard.html",
        users=users,
        predictions=predictions,
        accuracy=accuracy,
        soil=soil
    )


# ---------------- PREDICT ----------------

@app.route("/predict", methods=["GET","POST"])
def predict():

    if request.method == "POST":

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temp = float(request.form["temp"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])
        soil_moisture = float(request.form["soil_moisture"])

        prediction = model.predict([[N,P,K,temp,humidity,ph,rainfall,soil_moisture]])

        crop = encoder.inverse_transform(prediction)[0]

        predictions_collection.insert_one({
            "user_id":session["user_id"],
            "crop":crop,
            "soil_moisture":soil_moisture,
            "date":datetime.now()
        })

        return render_template(
            "result.html",
            result=crop,
            soil_moisture=soil_moisture
        )

    return render_template("predict.html")


# ---------------- API ----------------

@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.json

    N = data["N"]
    P = data["P"]
    K = data["K"]
    temp = data["temperature"]
    humidity = data["humidity"]
    ph = data["ph"]
    rainfall = data["rainfall"]
    soil_moisture = data["soil_moisture"]

    prediction = model.predict([[N,P,K,temp,humidity,ph,rainfall,soil_moisture]])

    crop = encoder.inverse_transform(prediction)[0]

    return jsonify({"crop":crop})


# ---------------- LOGOUT ----------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)
