import pickle

model = pickle.load(open("model.pkl","rb"))

def predict_crop(data):
    prediction = model.predict([data])
    return prediction[0]