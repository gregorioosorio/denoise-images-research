FROM tensorflow/tensorflow:latest-gpu

ADD . /tmp

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6

RUN pip install matplotlib \
    pip install scikit-image \
    pip install opencv-python
