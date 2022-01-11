FROM python:3.7.8-slim

COPY requirements/common.txt requirements/common.txt
RUN pip install -U pip && pip install -r requirements/common.txt

COPY ./api /app/api
COPY ./bin /app/bin
COPY ./models /app/models
COPY wsgi.py /app/wsgi.py
WORKDIR /app


EXPOSE 8080

ENTRYPOINT ["bash", "/app/bin/run.sh"]

