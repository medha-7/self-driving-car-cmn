FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive
    
WORKDIR /usr/bin/RL

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN git clone https://github.com/metadriverse/metadrive.git --single-branch && \
    cd metadrive && \
    pip install -e .

RUN pip install cupy-cuda11x

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 xvfb  -y


COPY model2.py /usr/bin/RL/model2.py
COPY train.py /usr/bin/RL/train.py
COPY train.py /usr/bin/RL/metadrive_test.py

WORKDIR /usr/bin/RL

RUN mkdir /usr/bin/RL/models
