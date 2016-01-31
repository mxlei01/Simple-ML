FROM ubuntu:14.04

#===========================Start Update Ubuntu================================
RUN apt-get -y update 
RUN apt-get -y upgrade 
RUN apt-get -y install apt-utils 
RUN apt-get -y update 
RUN apt-get -y upgrade
RUN apt-get -y install python3-pip
RUN apt-get -y install build-essential
RUN apt-get -y install python3-dev
RUN apt-get -y install libpq-dev
RUN apt-get -y install libblas-dev liblapack-dev
RUN apt-get -y install gfortran
RUN apt-get -y install git
#===========================End Update Ubuntu==================================

#=====Create a folder, and copy our application into the application folder====
RUN mkdir /application
COPY Application /application/Application
COPY .coveragerc /application/
COPY .git /application
#=====Create a folder, and copy our application into the application folder====

#===========================Installing Python Requirements=====================
COPY Python_Requirements /application/Python_Requirements
RUN pip3 install cython
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install -r /application/Python_Requirements/requirements.txt
#===========================Installing Python Requirements=====================
