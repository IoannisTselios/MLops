from google.cloud import storage
from datetime import datetime
import pickle
import torch


class SaveModel:
    def __init__(self, model, bucket_name):
        self.model = model
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def save(self):
        now = datetime.now()
        model_name = now.strftime("%Y-%m-%d-%H:%M:%S")
        blob_name = f"{model_name}.pth"
        blob = self.bucket.blob(blob_name)
        with blob.open("wb", ignore_flush=True) as f:
            torch.save(self.model.state_dict(), f)
        print(f"Model saved as {blob_name}.")
        return blob_name
