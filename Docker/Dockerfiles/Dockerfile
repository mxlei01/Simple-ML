FROM ubuntu:17.04

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
RUN apt-get -y install python-software-properties
RUN apt-get -y install software-properties-common
RUN apt-get -y install wget
#===========================End Update Ubuntu==================================

#=====Create a folder, and copy our application into the application folder====
RUN mkdir /application
COPY simpleml /application/simpleml
COPY Docker/Build_Scripts /application/Build_Scripts
COPY .coveragerc /application/
COPY .noserc /application/
COPY .git /application
#=====Create a folder, and copy our application into the application folder====

#=====Installing Python 3.5====================================================
RUN apt-get -y update
RUN apt-get -y install python3.6
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.6 get-pip.py
RUN apt-get -y install python3.6-dev
RUN apt-get -y install g++
RUN apt-get -y install libfreetype6-dev
RUN apt-get -y install python3.6-dev
#=====Installing Python 3.5====================================================

#===========================Installing Python Requirements=====================
COPY requirements.txt /application/requirements.txt
RUN pip3 install -r /application/requirements.txt
#===========================Installing Python Requirements=====================
