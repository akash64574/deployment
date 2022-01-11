import requests
import os


if __name__ == '__main__':
    url = 'http://127.0.0.1:8080/predict_api'
    path = os.path.dirname(__file__)
    test_json = os.path.join(path, 'test_cases/positive_case.json')
    headers = {'Content-Type':'application/json'}
    r = requests.post(url, data=open(test_json, 'rb'), headers=headers)
    print(r.status_code)
    print(r.json())
    exit(0)
