# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY dog_breed_identification/ dog_breed_identification/
COPY data/ data/

# This is 100% not the correct way to do it
COPY dog_breed_identification/config /usr/local/lib/python3.11/site-packages/dog_breed_identification/config

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
# For some reason numpy is not installed, for now install manually
RUN pip install numpy

ENTRYPOINT ["python", "-u", "dog_breed_identification/train_model.py"]