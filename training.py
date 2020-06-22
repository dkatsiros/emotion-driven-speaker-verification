import torch
from torch.nn import functional as F
import sys
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def train_and_validate(model,
                       train_loader,
                       valid_loader,
                       loss_function,
                       optimizer,
                       epochs,
                       cross_validation_epochs=5,
                       early_stopping=False):
    """
    Trains the given <model>.
    Then validates every <cross_validation_epochs>.
    Returns: <best_model> containing the model with best parameters.
    """

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
                           loss_function, optimizer)
        # Store statistics for later usage
        all_train_loss.append(train_loss)

        # ====== VALIDATION HERE ======
        if epoch % CROSS_VALIDATION_EPOCHS == 0:
            valid_loss = validate(epoch, valid_loader,
                                  model, loss_function)
            # Store model
            models.append(deepcopy(model))

            # Store statistics for later usage
            all_valid_loss.append(valid_loss)

        # Make sure enough epochs have passed
        if epoch < 4 * CROSS_VALIDATION_EPOCHS:
            continue

        # Remove first model when adding one new
        models = models[1:-1]
        best_model = model

        # Early stopping enabled?
        if early_stopping is False:
            continue
        # If enabled do everything needed
        STOP = True
        for i in range(2, 5):
            if valid_loss < all_train_loss[-i]:
                STOP = False
                break
        # Actually do early stopping
        if STOP is True:
            # But first save the model
            idx = np.argmax(all_valid_loss)
            best_model = models[idx]
            break

    return best_model, all_train_loss, all_valid_loss


def train(_epoch, dataloader, model, loss_function, optimizer):
    # Set model to train mode
    model.train()
    training_loss = 0.0

    # obtain the model's device ID
    device = next(model.parameters()).device

    # Iterate the batch
    for index, batch in enumerate(dataloader, 1):

        # Split the contents of each batch[i]
        try:
            # LSTM
            inputs, labels, lengths = batch
        except ValueError:
            # CNN
            inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass: y' = model(x)
        try:
            y_preds = model.forward(inputs, lengths)
        except:
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


def validate(_epoch, dataloader, model, loss_function):
    """Validate the model."""

    # Put model to evalutation mode
    model.eval()

    with torch.no_grad():
        valid_loss = 0

        for index, batch in enumerate(dataloader, 1):

            # Get the sample
            inputs, labels, lengths = batch

            # Forward through the network
            y_pred = model.forward(inputs, lengths)

            # Compute loss
            loss = loss_function(y_pred, labels.type('torch.LongTensor'))

            # Add validation loss
            valid_loss += loss.data.item()

        # Print some stats
        print(f'\nValidation loss at epoch {_epoch} : {round(valid_loss, 4)}')

    return valid_loss / len(dataloader)


def test(model, dataloader):
    """
    Tests a given model.
    Returns an array with predictions and an array with labels.
    """

    correct = 0
    # Create empty array for storing predictions and labels
    y_pred = []
    y_true = []
    for index, batch in enumerate(dataloader, 1):
        # Split each batch[index]
        inputs, labels, lengths = batch

        # Forward through the network
        out = model.forward(inputs, lengths)

        # Predict the one with the maximum probability
        predictions = F.softmax(out, dim=-1).argmax(dim=-1)

        # Save predictions
        y_pred.append(predictions.data.numpy())
        y_true.append(labels.data.numpy())

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

