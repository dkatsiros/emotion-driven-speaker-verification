import os
import sys
import time
import math
from inspect import getsource
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
# Report metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from utils import emodb
from utils import iemocap
from utils import voxceleb
from utils.early_stopping import EarlyStopping

from plotting.metrics import plot_confusion_matrix
from core.config import PLOTS_FOLDER, REPORTS_FOLDER
from core import config


def train(_epoch, dataloader, model, loss_function, optimizer, writer, cnn=False, *args, **kwargs):
    # Set model to train mode
    model.train()
    training_loss = 0.0
    correct = 0

    # obtain the model's device ID
    device = next(model.parameters()).device

    # Iterate the batch
    for index, batch in enumerate(dataloader, 1):

        # Split the contents of each batch[i]
        inputs, labels, lengths = batch

        inputs = inputs.to(device)
        labels = labels.type('torch.LongTensor').to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass: y' = model(x)
        if cnn is False:
            y_pred = model.forward(inputs, lengths)
        else:
            # We got a CNN
            # Add a new axis for CNN filter features, [z-axis]
            inputs = inputs[:, np.newaxis, :, :]
            y_pred = model.forward(inputs)

        # Compute loss: L = loss_function(y', y)
        loss = loss_function(y_pred, labels)

        labels_cpu = labels.detach().clone().to('cpu').numpy()
        # Get accuracy
        correct += sum([int(a == b)
                        for a, b in zip(labels_cpu,
                                        np.argmax(y_pred.detach().clone().to('cpu').numpy(), axis=1))])
        # Backward pass: compute gradient wrt model parameters
        loss.backward()

        # Update weights
        optimizer.step()

        # Add loss to total epoch loss
        training_loss += loss.data.item()

        # print statistics
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    accuracy = correct/len(dataloader.dataset) * 100
    writer.add_scalar("Accuracy/train", accuracy, _epoch)
    return training_loss / len(dataloader.dataset)


def validate(_epoch, dataloader, model, loss_function, writer, cnn=False, *args, **kwargs):
    """Validate the model."""

    # Put model to evalutation mode
    model.eval()

    correct = 0

    # obtain the model's device ID
    device = next(model.parameters()).device

    with torch.no_grad():
        valid_loss = 0

        for index, batch in enumerate(dataloader, 1):

            # Get the sample
            inputs, labels, lengths = batch

            # Transfer to device
            inputs = inputs.to(device)
            labels = labels.type('torch.LongTensor').to(device)

            # Forward through the network
            if cnn is False:
                y_pred = model.forward(inputs, lengths)
            else:
                # We got CNN
                # Add a new axis for CNN filter features, [z-axis]
                inputs = inputs[:, np.newaxis, :, :]
                y_pred = model.forward(inputs)

            labels_cpu = labels.detach().clone().to('cpu').numpy()
            # Get accuracy
            correct += sum([int(a == b)
                            for a, b in zip(labels_cpu,
                                            np.argmax(y_pred.detach().clone().to('cpu').numpy(), axis=1))])

            # Compute loss
            loss = loss_function(y_pred, labels)

            # Add validation loss
            valid_loss += loss.data.item()

        # Print some stats
        print(
            f'\nValidation loss at epoch {_epoch} : {round(valid_loss/len(dataloader.dataset), 4)}')

        accuracy = correct / len(dataloader.dataset) * 100
        writer.add_scalar("Accuracy/validation", accuracy, _epoch)

    return valid_loss / len(dataloader.dataset)


def train_and_validate(model,
                       train_loader,
                       valid_loader,
                       loss_function,
                       optimizer,
                       epochs,
                       cnn=False,
                       early_stopping=False,
                       valid_freq=5,
                       checkpoint_freq=config.CHECKPOINT_FREQ,
                       train_func=train,
                       validate_func=validate,
                       writer=SummaryWriter()):
    """
    Trains the given `model`.
    Then validates every `valid_freq`.
    Returns: `best_model` containing the model with best parameters.
    """

    # obtain the model's device ID
    device = next(model.parameters()).device

    # Store models
    best_model = None

    # Early stopping
    if early_stopping is not False:
        modelpath = os.path.join(config.CHECKPOINT_FOLDER, config.MODELNAME)
        early_stopping = EarlyStopping(patience=config.PATIENCE,
                                       delta=config.DELTA,
                                       path=modelpath)

    # ========= TRAIN & VALIDATION ================
    for epoch in range(1, epochs + 1):

        # Checkpoint
        if epoch % checkpoint_freq == 0:
            modelname = f"{config.CHECKPOINT_MODELNAME}_{epoch}_epoch_checkpoint.pt"
            modelpath = os.path.join(config.CHECKPOINT_FOLDER, modelname)
            torch.save(model.eval().cpu(), modelpath)
            model.to(device)

        # ===== Training HERE =====
        train_loss = train_func(epoch, train_loader, model,
                                loss_function, optimizer, cnn=cnn, writer=writer)
        # log training results
        writer.add_scalar("Loss/train", train_loss, epoch)
        mesg = f"{time.ctime()}\tEpoch:{epoch}\tTraining Loss:{train_loss}\n"
        logging.info(mesg)

        # ====== VALIDATION HERE ======
        if epoch % valid_freq == 0:
            valid_loss = validate_func(epoch, valid_loader,
                                       model, loss_function,
                                       cnn=cnn, writer=writer)

            # Store statistics for later usage
            writer.add_scalar("Loss/validation", valid_loss, epoch)

            # logging on file
            if config.LOGGING is True:
                # with open(config.LOG_FILE, mode='a') as file:
                # file.write(mesg)
                mesg = f"{time.ctime()}\tEpoch:{epoch}\t Validation Loss:{valid_loss}\n"
                logging.info(mesg)

            # Early Stopping
            if early_stopping is not False:
                # Move model to CPU
                early_stopping(val_loss=valid_loss, model=model)
                # Back to GPU
                model.to(device)
                if early_stopping.early_stop is True:
                    # Remove unused data from GPU
                    del model
                    # Load best stored model, so far
                    best_model = early_stopping.restore_best_model().to(device)
                    print(f"Restored best model from {early_stopping.path}")
                    writer.flush()
                    return best_model, epoch

    print(f'\nTraining exited normally at epoch {epoch}.')
    # Remove unnessesary model
    best_model = model.to(device)
    del model
    writer.flush()
    return best_model, epoch


def test(model, dataloader, cnn=False):
    """
    Tests a given model.
    Returns an array with predictions and an array with labels.
    """
    # obtain the model's device ID
    device = next(model.parameters()).device

    # Create empty array for storing predictions and labels
    y_pred = []
    y_true = []
    for index, batch in enumerate(dataloader, 1):
        # Split each batch[index]
        inputs, labels, lengths = batch

        # Transfer to device
        inputs = inputs.to(device)
        labels = labels.type('torch.LongTensor').to(device)
        # print(f'\n labels: {labels}')

        # Forward through the network
        if cnn is False:
            out = model.forward(inputs, lengths)
        else:
            # Add a new axis for CNN filter features, [z-axis]
            inputs = inputs[:, np.newaxis, :, :]
            out = model.forward(inputs)
        # Predict the one with the maximum probability
        # predictions = F.softmax(out, dim=-1).argmax(dim=-1)
        # print(f'\nout:{out}')
        predictions = torch.argmax(out, -1)
        # print(f'\nprediction: {predictions}')
        # Save predictions
        y_pred.append(predictions.cpu().data.numpy())
        y_true.append(labels.cpu().data.numpy())

    # Get metrics
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()

    return y_pred, y_true


def results(model, optimizer, loss_function,
            y_pred, y_true,
            epochs, timestamp,
            dataset):
    """Prints model details"""

    # Print metrics
    f1_metric = f1_score(y_true, y_pred, average='macro')
    print(f'f1 score: {round(f1_metric,4)}')
    acc = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {round(acc,4)}')

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    if dataset == "IEMOCAP":
        classes = iemocap.get_classes(n_classes=4)
    elif dataset == "EMODB":
        classes = emodb.get_classes()
    else:
        classes = iemocap.get_classes(n_classes=4)
    cnf_mtrx_filename = f'{model.__class__.__name__}_{epochs}_{timestamp}_confusion_matrix.png'
    plot_confusion_matrix(cm=conf_matrix, classes=classes,
                          filename=cnf_mtrx_filename)
    model_classification_report = classification_report(
        y_true=y_true, y_pred=y_pred, target_names=classes)

    # Raw model
    forward_raw = getsource(model.forward)
    # Save metrics
    filename = f'{model.__class__.__name__}_{epochs}_{timestamp}.md'
    with open(os.path.join(REPORTS_FOLDER, filename), mode='w') as f:
        data = f"""
# REPORT:
# {filename}
# Dataset
# {dataset}
# Model details:
```python
{model}
```
# Optimizer
```python
{optimizer}
```
# Loss function
```python
{loss_function}
```
# Metrics:
f1-macro-score = {round(f1_metric,4)}
acc = {round(acc,4)}
# Classification Report:
```python
{model_classification_report}
```
# Confusion matrix
<img src='../plots/{cnf_mtrx_filename}'>
# Forward
```python
{forward_raw}
```
    """
        f.write(data)


def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def overfit_batch(model,
                  train_loader,
                  loss_function,
                  optimizer,
                  epochs,
                  train_func=train,
                  cnn=False):
    """
    Trains a <model> with a given batch
    """
    import torch
    # obtain the model's device ID
    device = next(model.parameters()).device

    # Store losses, models
    all_train_loss = []

    # Get first batch
    first_batch = next(iter(train_loader))
    batch_dims = len(first_batch.size())
    # Create new train loader with using len(dataloader)*first batch
    train_loader = first_batch.repeat(
        (len(train_loader), *[1]*batch_dims))
    # Iterate for epochs
    for epoch in range(1, epochs + 1):

        # ===== Training HERE =====
        train_loss = train_func(epoch, train_loader, model,
                                loss_function, optimizer, cnn=cnn)
        # Store statistics for later usage
        all_train_loss.append(train_loss)
        print(f'\nEpoch {epoch} loss: {train_loss}')
        logging.info(f'\nEpoch {epoch} loss: {train_loss}')

    return model, all_train_loss, epoch


def deterministic_model(deterministic=False):
    """Set randomness to zero."""
    import torch
    import numpy as np
    import random
    print(f"Deterministic Model: {deterministic}")
    if deterministic is True:
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
