FROM python:3.8.17 AS builder
COPY requirements.txt /tmp

RUN apt update && apt upgrade -y
RUN apt install build-essential
RUN pip3 install --default-timeout=100 -r /tmp/requirements.txt

FROM builder AS app
COPY stock-prediction /stock-prediction
WORKDIR /stock-prediction/app
CMD ["python3", "stockPrediction.py"]