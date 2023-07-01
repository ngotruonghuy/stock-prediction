**Build Docker Image**
> docker build . -t stock-prediction:1.0

**Run Docker Image**
> docker run -d --name stock-prediction -p 8050:8050 stock-prediction:1.0