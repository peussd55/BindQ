FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

#FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

#ENV TZ=Europe/Berlin
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
#RUN apt install -y tzdata
#RUN apt-get -y update && apt-get -y install software-properties-common \
#&& add-apt-repository ppa:deadsnakes/ppa && apt install -y python3.10

RUN apt-get install -y python3 python3-pip git wget
 
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN python3 -m pip install --upgrade pip 

RUN pip install gradio==3.46.0
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python3", "app.py"]