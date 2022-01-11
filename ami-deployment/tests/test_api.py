import pytest

import json
from api.predictor import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_health(client):
    assert client.get("/health").status_code == 200


def test_predict(client):
    test_json = 'tests/test_cases/positive_case.json'
    res = client.post("/predict_api", data=open(test_json, 'rb'))
    assert res.status_code == 200

    probability = json.loads(res.data)['ModelRes']['Data'][0]['Value']
    assert probability == '83.18%'

