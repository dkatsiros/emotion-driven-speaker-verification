import torch
from torch.nn import functional as F
import sys
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Report metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from utils.emodb import get_classes
from sklearn.metrics import f1_score, accuracy_score
from plotting.metrics import plot_confusion_matrix


def train_and_validate(model,
                       train_loader,
                       valid_loader,
                       loss_function,
                       optimizer,
                       epochs,
                       cnn=False,
                       cross_validation_epochs=5,
                       early_stopping=False):
    """
    Trains the given <model>.
    Then validates every <cross_validation_epochs>.
    Returns: <best_model> containing the model with best parameters.
    """

    # obtain the model's device ID
    device = next(model.parameters()).device

    print(next(iter(train_loader)))

    EPOCHS = epochs
    CROSS_VALIDATION_EPOCHS = cross_validation_epochs
    best_model = model

    # Store losses, models
    all_train_loss = []
    all_valid_loss = []
    models = []

    # Iterate for EPOCHS
    for epoch in range(1, EPOCHS + 1):

        # ===== Training HERE =====
        train_loss = train(epoch, train_loader, model,
                           loss_function, optimizer, cnn=cnn)
        # Store statistics for later usage
        all_train_loss.append(train_loss)

        # ====== VALIDATION HERE ======
        if epoch % CROSS_VALIDATION_EPOCHS == 0:
            valid_loss = validate(epoch, valid_loader,
                                  model, loss_function, cnn=cnn)
            # Store model but on cpu
            models.append(deepcopy(model).to('cpu'))
            # Remove first model when adding one new if len(models)>5
            if len(models) == 6:
                models = models[1:-1]
            # Store statistics for later usage
            all_valid_loss.append(valid_loss)

        # Make sure enough epochs have passed
        if epoch < 4 * CROSS_VALIDATION_EPOCHS:
            continue

        best_model = model

        # Early stopping enabled?
        if early_stopping is False:
            continue
        # If enabled do everything needed
        STOP = True
        # Check last 4 validations
        for i in range(1, 4):
            # If at least one val_loss_new < val_loss_older dont stop
            if all_valid_loss[-i] < all_valid_loss[-i-1]:
                STOP = False
                break
        # Actually do early stopping
        if STOP is True:
            # Reset last model
            print('Resetting to previous model! ..\n')
            best_model = models[-3].to(device)
            break

    return best_model, all_train_loss, all_valid_loss, epoch


def train(_epoch, dataloader, model, loss_function, optimizer, cnn=False):
    # Set model to train mode
    model.train()
    training_loss = 0.0

    # obtain the model's device ID
    device = next(model.parameters()).device

    # Iterate the batch
    for index, batch in enumerate(dataloader, 1):

        # Split the contents of each batch[i]
        inputs, labels, lengths = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass: y' = model(x)
        if cnn is False:
            y_preds = model.forward(inputs, lengths)
        else:
            # We got a CNN
            # Add a new axis for CNN filter features, [z-axis]
            inputs = inputs[:, np.newaxis, :, :]
            y_preds = model.forward(inputs)

        # Compute loss: L = loss_function(y', y)
        loss = loss_function(y_preds, labels)

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

    # Return loss
    return training_loss / len(dataloader)


def validate(_epoch, dataloader, model, loss_function, cnn=False):
    """Validate the model."""

    # Put model to evalutation mode
    model.eval()

    # obtain the model's device ID
    device = next(model.parameters()).device

    with torch.no_grad():
        valid_loss = 0

        for index, batch in enumerate(dataloader, 1):

            # Get the sample
            inputs, labels, lengths = batch
            # Transfer to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward through the network
            if cnn is False:
                y_pred = model.forward(inputs, lengths)
            else:
                # We got CNN
                # Add a new axis for CNN filter features, [z-axis]
                inputs = inputs[:, np.newaxis, :, :]
                y_pred = model.forward(inputs)

            # Compute loss
            loss = loss_function(y_pred, labels)

            # Add validation loss
            valid_loss += loss.data.item()

        # Print some stats
        print(f'\nValidation loss at epoch {_epoch} : {round(valid_loss, 4)}')

    return valid_loss / len(dataloader)


def test(model, dataloader, cnn=False):
    """
    Tests a given model.
    Returns an array with predictions and an array with labels.
    """
    # obtain the model's device ID
    device = next(model.parameters()).device

    correct = 0
    # Create empty array for storing predictions and labels
    y_pred = []
    y_true = []
    for index, batch in enumerate(dataloader, 1):
        # Split each batch[index]
        inputs, labels, lengths = batch

        # Transfer to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward through the network
        if cnn is False:
            out = model.forward(inputs, lengths)
        else:
            # Add a new axis for CNN filter features, [z-axis]
            inputs = inputs[:, np.newaxis, :, :]
            out = model.forward(inputs)
        # Predict the one with the maximum probability
        predictions = F.softmax(out, dim=-1).argmax(dim=-1)

        # Save predictions
        y_pred.append(predictions.cpu().data.numpy())
        y_true.append(labels.cpu().data.numpy())

    # Get metrics
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()

    return y_pred, y_true


def results(model, train_loss, valid_loss, y_pred, y_true, epochs, cv=5):
    """Prints the results of training. Also saves some plots."""
    # Plots
    fig = plt.figure(figsize=(10, 10))
    plt.ylabel('Train - Validation Loss')
    plt.xlabel('Epoch')
    try:
        plt.plot(list(range(1, epochs + 1)), train_loss, color='r')
        valid_loss_for_plot = np.array([[l]*cv for l in valid_loss]).flatten()
        plt.plot(list(range(1, epochs + 1)), valid_loss_for_plot, color='b')
        plt.savefig(
            f'plotting/plots/{model.__class__.__name__}_train_valid_loss.png')
    except Exception as e:
        print('\Exception raised while creating plot: {e}')

    # Print metrics
    f1_metric = f1_score(y_true, y_pred, average='macro')
    print(f'f1 score: {round(f1_metric,4)}')
    acc = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {round(acc,4)}')


def results(model, train_loss, valid_loss, y_pred, y_true, epochs, timestamp, cv=5):
    """Prints the results of training. Also saves some plots."""
    # Plots
    fig = plt.figure(figsize=(10, 10))
    plt.ylabel('Train - Validation Loss')
    plt.xlabel('Epoch')
    try:
        plt.plot(list(range(1, epochs + 1)), train_loss,
                 color='r', label='Training')
        vld_values = np.array([[l]*cv for l in valid_loss]).flatten()
        # Add some missing values below
        vld_values = np.concatenate(
            (vld_values, np.array([valid_loss[-1]]*(epochs % cv))))

        plt.plot(list(range(1, epochs + 1)), vld_values,
                 color='b', label='Validation')
        plt.legend()
        plot_filename = f'train_valid_loss_{model.__class__.__name__}_{epochs}_{timestamp}.png'
        plt.savefig(
            f'plotting/plots/{plot_filename}')
    except Exception as e:
        print(f'Exception raised while creating plot: {e}')
    finally:
        # Just in case plotting fails. We need the name below
        plot_filename = f'train_valid_loss_{model.__class__.__name__}_{epochs}_{timestamp}.png'

    # Print metrics
    f1_metric = f1_score(y_true, y_pred, average='macro')
    print(f'f1 score: {round(f1_metric,4)}')
    acc = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {round(acc,4)}')

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    classes = get_classes()
    cnf_mtrx_filename = f'{model.__class__.__name__}_{epochs}_{timestamp}_confusion_matrix.png'
    plot_confusion_matrix(cm=conf_matrix, classes=classes,
                          filename=cnf_mtrx_filename)
    model_classification_report = classification_report(
        y_true=y_true, y_pred=y_pred, target_names=classes)
    # Save metrics
    filename = f'{model.__class__.__name__}_{epochs}_{timestamp}.md'
    with open(f'plotting/reports/{filename}', mode='w') as f:
        data = f"""
# REPORT:
### {filename}
f1-macro-score = {f1_metric}
acc = {acc}
```
{model_classification_report}
```
<img src='../plots/{plot_filename}'>
<img src='../plots/{cnf_mtrx_filename}'>
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
