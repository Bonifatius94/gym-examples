FROM tensorflow/tensorflow:2.9.1-gpu

ADD ./requirements.txt ./
RUN pip install -r requirements.txt

ADD ./training.py ./
ENTRYPOINT ["python", "training.py"]
