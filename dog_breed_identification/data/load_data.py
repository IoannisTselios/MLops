from google.cloud import storage
import hydra
import os
import io
import torch

class LoadData:
    def load(blob_name, train_config):
        storage_client = storage.Client()
        bucket_name = train_config.data_bucket_name
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # rb important otherwise it will be read as string and fails with some character encoding error
        with blob.open("rb") as f:
            data = f.read()
        return torch.load(io.BytesIO(data))

if __name__ == "__main__":
    print(LoadData())