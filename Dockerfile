# Pull latest CentOS image
FROM centos:latest

# Creator
MAINTAINER Sam Potter <spotter1642@gmail.com>

# Define a user
RUN useradd -u 2000 -m test

# Install wget
RUN yum -y update
RUN yum -y install wget

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p $HOME/miniconda

# Update path
RUN export PATH="$HOME/minconda/bin:$PATH"
RUN hash -r

# Install conda
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN conda info -a
RUN conda create -q -n test-env python=3.6 pytest pytest-cov
RUN source activate test-env
RUN conda install numpy