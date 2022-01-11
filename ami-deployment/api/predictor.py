from flask import Flask, Response, request
import json
from . import score

app = Flask(__name__)

# Load model
score.init()


@app.route("/health")
def index():
    return Response("OK!", status=200)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    For direct API calls trought request
    """

    data = request.get_json(force=True)
    out_json = score.run(json.dumps(data))

    return out_json


if __name__ == '__main__':
    app.run(debug=True)
