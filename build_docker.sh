#!/bin/bash

docker rmi -f mytensorflow:latest
docker build --rm -f Dockerfile -t mytensorflow .