# syntax=docker/dockerfile:1
# Pull base image
FROM ubuntu:20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set work directory
WORKDIR /prj
COPY . .
RUN apt-get update && apt-get install -y python3 python3-pip libgdal-dev git vim
RUN pip install -r requirements.txt
