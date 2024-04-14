import click
import torch
from torch.utils.data import DataLoader
import hydra
import wandb
import pandas as pd
from dog_breed_identification.models.model import Model
from dog_breed_identification.data.load_data import LoadData
from dog_breed_identification.models.save_model import SaveModel
import os
import omegaconf

# Load config
hydra.initialize(config_path="config", version_base=None)
train_config = hydra.compose(config_name="train_config")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = train_config.data_path

# Load model
model = Model()
model.to(device)
model.train()


def train(num_epochs, learning_rate, batch_size, model_name):
    # Load data
    if os.path.exists(f"{dataset_path}/train.pt"):
        train = torch.load("data/processed/train.pt")
    else:
        train = LoadData.load(f'{dataset_path}/train.pt', train_config)
    # Create dataloaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        overall_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device).float()
            y = y.to(device).float()
            preds = model(x)
            loss = loss_fn(preds, y)
            overall_loss += loss.item()

            predicted_index = torch.argmax(preds.data, 1)
            correct_index = torch.argmax(y, 1)

            correct_on_batch = (predicted_index == correct_index).sum().item()

            correct += correct_on_batch
            total += batch_size

            if batch_idx % 1 == 0:
                print(
                    f"Batch Index: {batch_idx} Loss: {loss.item() / y.size(0)}")
                wandb.log({"Training Loss (batch)": loss.item() / y.size(0)})
                wandb.log({"Training Accuracy (batch)": 100 *
                          correct_on_batch / y.size(0)})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} complete! \t Average Loss: {overall_loss / ((batch_idx + 1) * batch_size)} \t Accuracy: {100 * correct / total}")

        wandb.log({"Epoch": epoch + 1})
        wandb.log({"Average Loss (epoch)": overall_loss /
                  ((batch_idx + 1) * batch_size)})
        wandb.log({"Training Accuracy (epoch)": 100 * correct / total})

    print("Training complete, saving model...")

    # Save model
    print(train_config.model_bucket_name)
    model_saver = SaveModel(model, train_config.model_bucket_name)
    model_name = model_saver.save()
    wandb.run.summary({'Model Name': model_name})


def evaluate(batch_size):

    print("Evaluating model...")
    # Load data
    if os.path.exists(f"{dataset_path}/val.pt"):
        val = torch.load(f"{dataset_path}/val.pt")
    else:
        val = LoadData.load(f'{dataset_path}/val.pt', train_config)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device).float()
            y = y.to(device).float()
            preds = model(x)
            total += y.size(0)

            predicted_index = torch.argmax(preds.data, 1)
            correct_index = torch.argmax(y, 1)
            correct += (predicted_index == correct_index).sum().item()

    labels = pd.read_csv('data/processed/breeds.csv', names=["id", "breed"])

    for i in range(5):
        correct_breed = labels[labels["id"] ==
                               correct_index[i].item()]["breed"].values[0]
        predicted_breed = labels[labels["id"] ==
                                 predicted_index[i].item()]["breed"].values[0]

        wandb.log({
            "Examples": wandb.Image(x[i], caption=f"Predicted: {predicted_breed}, Correct: {correct_breed}"),
        })

    accuracy = 100 * correct / total

    print(f"Accuracy (validation): {accuracy}")
    wandb.run.summary({'Accuracy (validation)': accuracy})


@click.command()
@click.option('--num_epochs', default=train_config.epochs, type=int, help='Set the num_epochs')
@click.option('--learning_rate', default=train_config.lr, type=float, help='Set the learning_rate')
@click.option('--batch_size', default=train_config.batch_size, type=int, help='Set the batch_size')
@click.option('--model_name', default=train_config.name, type=str, help='Set the model file name')
def main(num_epochs, learning_rate, batch_size, model_name):
    if num_epochs != train_config.epochs:
        train_config.epochs = num_epochs

    if learning_rate != train_config.lr:
        train_config.lr = learning_rate

    if batch_size != train_config.batch_size:
        train_config.batch_size = batch_size

    if model_name != train_config.name:
        train_config.name = model_name

    config = omegaconf.OmegaConf.to_container(
        train_config, resolve=True, throw_on_missing=True
    )
    wandb.init(project="dog_breed_identification", config=config)

    train(num_epochs, learning_rate, batch_size, model_name)
    evaluate(batch_size)


if __name__ == '__main__':
    main()
