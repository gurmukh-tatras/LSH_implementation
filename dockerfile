FROM  tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install sklearn
RUN pip3 install tensorflow-gpu
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