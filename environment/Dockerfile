# A great resource: https://github.com/jupyter/docker-stacks
FROM jupyter/datascience-notebook:latest 

USER root
COPY apt-requirements.txt apt-requirements.txt 
COPY requirements.txt pip-requirements.txt 
 
RUN pip install -r pip-requirements.txt && \
apt-get update && \ 
cat apt-requirements.txt | xargs sudo apt-get install -y
