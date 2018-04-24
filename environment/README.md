# Installation Instructions 

## Local Installation (not recommended) 

We assume you have already have Jupyter and a python 3.6.2 kernel running. 

You can install the `pip` dependencies using the following command:

```
pip install -r requirements.txt
```

Make sure `graphviz` is installed on your system. If you are using a debian/ubuntu
machine, run:

```
apt-get install -y graphviz
```

## Use Docker 

### Build pre-configured Image 

You can build the Docker image necessary to run the webinar: 

```
docker build -t network-intrusion-webinar . 
```

### Pull Image from Dockerhub

Alternatively, you can pull the image from dockerhub. Run this command : 

```
docker pull jrgauthier01/network-intrusion-webinar
```

## Run Container 

The first thing you want to do is change ownership of the `network-intrusion-webinar` git repo to the 
within-container notebook user or group. This will allow you to mount the host `network-intrusion-webinar` 
directory as a folder on the container. Run the command : 

```bash 
chown 1000 /path/to/network-intrusion-webinar-git-repo/
``` 

Then run the following docker command: 

```bash
docker run -it --rm -v /path/to/network-intrusion-webinar-git-repo:/home/joyvan/work -p 8888:8888 network-intrusion-webinar:latest
```

this will open a jupyter window. If you go in the `work/` folder, you will see the content of the 
network-intrusion-webinar git repo. 

