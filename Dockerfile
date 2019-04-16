# Pull latest CentOS image
FROM continuumio/miniconda3

# Creator
MAINTAINER Sam Potter <spotter1642@gmail.com>

# Define the shell
SHELL ["/bin/bash", "-c"]

# Conda env
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# Copy code in
COPY . /code
