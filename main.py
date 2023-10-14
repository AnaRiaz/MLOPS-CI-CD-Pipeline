from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
data = load_iris()
X, y = data.data, data.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    measurements = request.json['measurements']
    prediction = model.predict([measurements])
    return jsonify({'species': data.target_names[prediction[0]]})

if __name__ == '__main__':
    app.run(debug=True)
