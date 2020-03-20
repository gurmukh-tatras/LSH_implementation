FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y python-pip

RUN pip install numpy
RUN pip install pandas
RUN pip install sklearn
# RUN pip install tensorflow-gpu
# RUN pip install celery
# RUN pip install redis
# RUN pip install pillow
# RUN pip install tensorflow
# RUN pip install keras
# RUN pip install scikit-image
# RUN pip install protobuf
# RUN pip install imutils

WORKDIR /app/
COPY . .
RUN python main.py