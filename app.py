
from flask import Flask, request, render_template
import joblib  # If using scikit-learn

app = Flask(__name__)

# Load your trained machine learning model (e.g., scikit-learn)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return f'Predicted Output: {prediction[0]}'

if __name__ == '__main__':
    app.run()