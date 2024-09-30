FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y g++

EXPOSE 8000

ADD req.txt .
RUN pip install -r req.txt

WORKDIR /usr/src/predict

ADD graph_v3.pkl .
ADD repo4eu.py .
ADD model_version_3.1_mashup.pth .
ADD nodes_v3.1.pkl .
ADD app.py .

CMD ["shiny", "run", "app.py", "--host=0.0.0.0"]
