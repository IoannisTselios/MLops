import io
import logging
from fastapi import FastAPI, UploadFile, BackgroundTasks, File
import timm
import torch
from google.cloud import storage
import torchvision
import pandas as pd
from typing import List
from fastapi.responses import HTMLResponse
from transformers import CLIPProcessor, CLIPModel
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from torch.utils.data import DataLoader
from evidently.report import Report
from PIL import Image
from io import BytesIO
import torchvision.transforms.functional as F
import traceback


def get_breeds():
    BUCKET_NAME = "dtu-mlops-data-bucket"
    BREEDS_FILE = "data/processed/breeds.csv"

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(BREEDS_FILE)

    blob.download_to_filename(f"src/breeds.csv")


def get_params():
    BUCKET_NAME = "dog-breed-identification-model"
    MODEL_FILE = "exp2.pth"

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(MODEL_FILE)

    with open(MODEL_FILE, 'wb') as file_obj:
        blob.download_to_file(file_obj)
    params = torch.load(MODEL_FILE, map_location=torch.device('cpu'))

    return params


def get_model():
    params = get_params()
    model = timm.create_model('mobilenetv3_large_100', num_classes=120)
    model.load_state_dict(params)
    model.eval()

    return model


def predict_one(imagefile, filename):
    try:
        contents = imagefile.read()
        with open("image.png", "wb") as f:
            f.write(contents)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((200, 200))
        ])
        print("here1")
        image = torchvision.io.read_image("image.png")
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.float()
        output = model(image)
        output = torch.softmax(output, dim=1)
        probability, breed_idx = torch.max(output, dim=1)
        predicted_breed = breeds[breeds["id"] ==
                                 breed_idx.item()]["breed"].values[0]

        return {
            "filename": filename,
            "probability": probability.item(),
            "breed_idx": breed_idx.item(),
            "breed": predicted_breed,
            "status": "success",
            "message": "Cute puppy detected"
        }
    except Exception as e:
        print(e)
        return {
            "filename": filename,
            "probability": -1,
            "breed_idx": -1,
            "breed": "-",
            "status": "error",
            "message": "There was an error predicting for this image. Perhaps the file is not an image or the format of the image is corrupted."
        }


# Function to upload a file to Google Cloud Storage
def upload_to_gcs(bucket_name, file_stream, destination_blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file_stream)


def extract_clip_features(images):
    # Process images and extract features using CLIP
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        img_features = clip_model.get_image_features(**inputs)
    return img_features.cpu().numpy()


def list_blobs_in_bucket(bucket_name, prefix=None):
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs if not blob.name.endswith('/')]


def download_image_from_gcs(bucket_name, blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    image_bytes = blob.download_as_bytes()
    return Image.open(BytesIO(image_bytes))


def download_train_data_from_gcs():
    # Get the bucket and blob
    bucket_name = 'dtu-mlops-data-bucket'
    blob_name = "data/processed/train.pt"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the file as bytes
    file_bytes = blob.download_as_bytes()
    return file_bytes


def convert_tensor_to_image(tensor):
    image = F.to_pil_image(tensor)
    return image


app = FastAPI()
storage_client = storage.Client()
GCS_BUCKET_NAME = 'dtu-mlops-input-images'
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
logging.basicConfig(level=logging.INFO)
model = get_model()
get_breeds()
breeds = pd.read_csv('src/breeds.csv', names=["id", "breed"])


@app.get("/")
async def root():
    return {  
        "message": "Hi"
    }


@app.post("/predict")
async def predict_images(background_tasks: BackgroundTasks, 
                         imagefiles: List[UploadFile] = File(...),):
    predictions = []
    upload_responses = []
    
    for imagefile in imagefiles:
        filename = imagefile.filename
        image_stream = io.BytesIO(await imagefile.read())
        
        # Perform prediction
        try:
            prediction = predict_one(image_stream, filename)
        except Exception as e:
            # Handle prediction error
            predictions.append({"error": f"Prediction failed for {imagefile.filename}: {str(e)}"})
            continue

        # Reset the file stream position for uploading
        image_stream.seek(0)

        # Construct the destination blob
        destination_blob_name = f"images/{imagefile.filename}"

        # Add the upload task to the background
        background_tasks.add_task(upload_to_gcs, GCS_BUCKET_NAME, image_stream, destination_blob_name)
        upload_responses.append(f"File '{imagefile.filename}' will be uploaded in the background")

        # Construct the GCS URL for the uploaded file
        gcs_url = f"gs://{GCS_BUCKET_NAME}/{destination_blob_name}"

        # Append the prediction result with the GCS URL
        prediction_result = {
            "image_url": gcs_url,
            **prediction
        }
        predictions.append(prediction_result)

    return predictions


@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring():
    try:
        reference_percentage = 0.001

        # Load the training data
        logging.info("Loading training data...")
        train = torch.load(BytesIO(download_train_data_from_gcs()))
        train_loader = DataLoader(train, batch_size=64, shuffle=True)
        total_batches = len(train_loader)
        batches_to_take = max(1, int(total_batches * reference_percentage))

        reference_images = []
        for i, data in enumerate(train_loader):
            images, _ = data
            reference_images.extend(images)
            if i + 1 == batches_to_take:
                break

        logging.info(f"Loaded {len(reference_images)} reference images.")

        # Process production images
        logging.info("Processing production images...")
        production_image_blobs = list_blobs_in_bucket(GCS_BUCKET_NAME, prefix='images/')
        production_images = [download_image_from_gcs(GCS_BUCKET_NAME, blob_name) for blob_name in production_image_blobs]

        # Extract features
        logging.info("Extracting features from images...")
        reference_features = extract_clip_features(reference_images)
        production_features = extract_clip_features(production_images)

        # Prepare DataFrames
        logging.info("Preparing DataFrames for analysis...")
        reference_df = pd.DataFrame(reference_features)
        production_df = pd.DataFrame(production_features)

        # Check if DataFrames are not empty
        if reference_df.empty or production_df.empty:
            logging.error("One of the DataFrames is empty. Aborting analysis.")
            return HTMLResponse(content="Error: DataFrames are empty.", status_code=500)

        reference_df.columns = [str(i) for i in range(reference_df.shape[1])]
        production_df.columns = [str(i) for i in range(production_df.shape[1])]

        # Generate report
        logging.info("Generating Evidently report...")
        # report = Report(metrics=[DataDriftPreset()])#, DataQualityPreset(), TargetDriftPreset()])
        data_drift_report = Report(
                                    metrics=[
                                        DataDriftPreset(),
                                    ]
                                )
                                
        data_drift_report.run(reference_data=reference_df.iloc[1:5],
                                current_data=production_df[1:5],
                                column_mapping=None,)

        # Save and read report
        report_html = 'monitoring.html'
        data_drift_report.save_html(report_html)

        with open(report_html, "r", encoding="utf-8") as f:
            html_content = f.read()
        logging.info("Done...")
        return HTMLResponse(content=html_content, status_code=200)

    except Exception as e:
        logging.error(f"An error occurred in monitoring endpoint: {e}")
        traceback.print_exc()
        return HTMLResponse(content=f"Error: {e}", status_code=500)