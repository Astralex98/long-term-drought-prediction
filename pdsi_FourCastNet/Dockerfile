# Pull base image
FROM ubuntu:20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app
RUN apt-get update && apt-get install -y python3 python3-pip libgdal-dev git vim
RUN pip install -r requirements.txt
RUN pip install --force-reinstall torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/
# Install GDAL (for preprocessing) 
ARG CPLUS_INCLUDE_PATH=/usr/include/gdal
ARG C_INCLUDE_PATH=/usr/include/gdal
RUN pip3 install gdal==$(gdal-config --version)
# dependencies for NVIDIA modulus
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --force pillow==9.5.0
# Copy project
COPY . /app/

