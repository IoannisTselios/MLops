import torch
import torchvision
import os
import pandas as pd
from dog_breed_identification.models.model import Model


def predict(model: torch.nn.Module, directory: str, return_label: bool = True) -> None:
    """Run prediction for a given model and folder of pictures with a .jpg extension.

    Args:
        model: model to use for prediction
        directory: path to folder with pictures to predict on
        return_label: whether to return the label of the predicted class in a string format or the index of the class

    Returns
        List of tuples (image, label) if return_label is True, else (image, index)

    """
    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    jpg_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    print(jpg_files)
    transform = torchvision.transforms.Resize((200, 200))

    if return_label:
        labels = pd.read_csv('data/processed/breeds.csv',
                             index_col=0, names=['breed'])

    output = []
    for jpg_file in jpg_files:
        image = torchvision.io.read_image(f'{directory}/{jpg_file}')
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.float()
        image = image.to(device)
        model.eval()
        preds = model(image)
        predicted_index = torch.argmax(preds.data, 1)
        if return_label:
            label = labels.iloc[predicted_index]["breed"].item()
            output.append((jpg_file, label))
        else:
            output.append((jpg_file, predicted_index))

    return output
