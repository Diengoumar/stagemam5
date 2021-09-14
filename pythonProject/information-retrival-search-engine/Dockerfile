############################################################
# Dockerfile to run a Django-based web application
# Based on an Ubuntu Image
############################################################

# Set the base image to use to Ubuntu
FROM ubuntu:16.04


# Set the file maintainer (your name - the file's author)
MAINTAINER oumar dieng


# Set env variables used in this Dockerfile (add a unique prefix, such as DOCKYARD)
# Local directory with project source
ENV DOCKYARD_SRC=informationRetrival
# Directory in container for all project files
ENV DOCKYARD_SRVHOME=/srv
# Directory in container for project source files
ENV DOCKYARD_SRVPROJ=/srv/informationRetrival

# Update the default application repository sources list
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python2.7 python-pip
#RUN apt-get install -y build-essential python-dev python-numpy \
#python-setuptools python-scipy libatlas-dev libatlas3-base python-matplotlib
RUN pip install --upgrade "pip < 21.0"
# Create application subdirectories
WORKDIR $DOCKYARD_SRVHOME
RUN mkdir media static logs
VOLUME ["$DOCKYARD_SRVHOME/media/", "$DOCKYARD_SRVHOME/logs/"]

# Copy application source code to SRCDIR
COPY $DOCKYARD_SRC $DOCKYARD_SRVPROJ

# Install Python dependencies
#RUN pip install -r informationRetrival/requirements.txt
RUN pip install Django==1.9.4
RUN pip install gunicorn==19.6.0
RUN pip install Whoosh==2.7.4
RUN pip install djangorestframework==3.6.2
RUN pip install scikit-learn==0.18
#RUN pip install scipy
#RUN pip install nltk
RUN pip install pymongo
#RUN pip install keras
RUN pip install tensorflow
RUN pip install --user --upgrade tensorflow-gpu
RUN pip install --user --upgrade tensorboard
RUN pip install keras==2.3.1
RUN pip install pandas
RUN pip install -U bert-serving-server bert-serving-client
RUN pip install joblib
RUN pip install "nltk == 3.4.5"
RUN pip install pillow

# Port to expose
EXPOSE 8000

# Copy entrypoint script into the image
WORKDIR $DOCKYARD_SRVPROJ
COPY ./docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
