# Step 1: Download the data from Kaggle (https://www.kaggle.com/competitions/dog-breed-identification/data)
# Step 2: Put it in the data/raw folder and unzip it
# Step 3: Enter 'make data' command in the terminal

import torch
import torchvision
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import TensorDataset
# from PIL import Image


def get_labels(labels):
    """Returns a list of ids and a tensor of labels"""
    ids = []
    label_tensor = []
    columns = labels.columns[2:]
    for rowindex in range(len(labels)):
        row_vector = []
        ids.append(labels.iloc[rowindex].iloc[0])
        for column in columns:
            row_vector.append(labels.iloc[rowindex].loc[column])
        label_tensor.append(torch.BoolTensor(row_vector))
    return ids, torch.stack(label_tensor)


def process_data():
    # Load the data and one-hot encode breed
    data_dir = "data/raw/dog-breed-identification"
    labels = pd.read_csv(f'{data_dir}/labels.csv')
    one_hot = pd.get_dummies(labels['breed'])
    labels = pd.concat([labels, one_hot], axis=1)
    train_ids, train_labels = get_labels(labels)

    # Split the data into train and test
    train_images = []
    test_images = []

    test_files = os.listdir(f'{data_dir}/test')

    transform = torchvision.transforms.Resize((200, 200))

    for train_id in train_ids:
        image = torchvision.io.read_image(f'{data_dir}/train/{train_id}.jpg')
        image = transform(image)
        train_images.append(image)

    for test_file in test_files:
        image = torchvision.io.read_image(f'{data_dir}/test/{test_file}')
        image = transform(image)
        test_images.append(image)

    train_images = torch.stack(train_images)
    test_images = torch.stack(test_images)

    # Split into train and validate
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_index, val_index = next(sss.split(train_images, train_labels))

    train = TensorDataset(train_images[train_index], train_labels[train_index])
    val = TensorDataset(train_images[val_index], train_labels[val_index])

    torch.save(train, 'data/processed/train.pt')
    torch.save(val, 'data/processed/val.pt')
    torch.save(test_images, 'data/processed/test.pt')

    print(f"Shape of train images: {train_images.shape}")

    breeds_list = list(one_hot.columns)
    breeds_df = pd.DataFrame(breeds_list)
    breeds_df.to_csv('data/processed/breeds.csv', index=True, header=False)


if __name__ == '__main__':
    process_data()
