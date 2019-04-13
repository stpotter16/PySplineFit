# Pull latest CentOS image
FROM centos:latest

# Creator
MAINTAINER Sam Potter <spotter1642@gmail.com>

# Define a user
RUN useradd -u 2000 -m test