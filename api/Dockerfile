FROM python:3.11
COPY ./requirements.txt /api/requirements.txt
COPY ./src /api/src

RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt
RUN pip install uvicorn python-multipart google-cloud-storage

WORKDIR /api

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]