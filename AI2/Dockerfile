FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set environment variables to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


# Update the apt package list and install any necessary packages (customize as needed)
RUN apt-get update && apt-get install -y \
    apt-utils
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    python3-dev \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /stage/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.12 get-pip.py

CMD ["/stage/entrypoint.sh"]
