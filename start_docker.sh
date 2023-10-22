#!/bin/bash

docker stop my_tf_container && docker rm my_tf_container
docker run --gpus all -it --name my_tf_container mytensorflow:latest bash
