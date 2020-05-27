import torch
from torch.nn import functional as F
import sys
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score


def fit(model, epochs, lr, loader, v_loader, L2=0, valTime=5, showVal=True, earlyStopping=False):
    """Train the model and also validate."""

    # Define loss function.
    loss_func = torch.nn.CrossEntropyLoss()
    # All loss
    train_loss_per_epoch = []
    valid_loss_per_epoch = []
    # Use Stochastic Gradient Descent of the optim package for parameter updates.
    # opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=L2)
    opt = torch.optim.Adadelta(
        model.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=L2)


    # A flag  for validation loss.
    old_valid_loss = -62
    old_net = model
    for epoch in range(epochs):

        ## TRAINING ---------------------
        # Initialize loss
        epoch_loss = 0
        # It is wise, according to the documentation, to put the network in
        # training mode before each batch is loaded.
        model.train()
        # iterate for inputs, labels, lengths
        for xb, yb, lb in loader:
            # Predict
            out = model.forward(xb, lb)
            # Get loss
            loss = loss_func(out, yb.type('torch.LongTensor'))
            # Add loss to current epoch
            epoch_loss += loss
            # Backpropagate the loss.
            loss.backward()
            # Update weights, bias.
            opt.step()
            # A new batch is ready to be loaded. Clean gradient memory!
            opt.zero_grad()
        print('Train loss at epoch', epoch, ':', float(epoch_loss))
        # Also store
        train_loss_per_epoch.append(float(epoch_loss))


        ## VALIDATION ---------------------------
        # At the end of each epoch, the network is put in evaluation mode.
        model.eval()
        # Will infer on the validation set (and maybe check for early       \
        # stopping) every earlyCheck epochs.
        if showVal and epoch % valTime == 0:
            # No reason to keep the gradient for the validation set.
            with torch.no_grad():
                valid_loss = 0
                for xb, yb, lb in v_loader:
                    out = model.forward(xb, lb)#.view(1, -1)
                    valid_loss += loss_func(out, yb.type('torch.LongTensor'))
                print('Validation loss at epoch',
                      epoch, ':', float(valid_loss))

                valid_loss_per_epoch.append(float(valid_loss))
                # Early stopping!
                if (earlyStopping and epoch >= 20 and
                    valid_loss_per_epoch[-1] > valid_loss_per_epoch[-2] and
                    valid_loss_per_epoch[-1] > valid_loss_per_epoch[-3] and
                    valid_loss_per_epoch[-1] > valid_loss_per_epoch[-4]):
                    # if the criterion is broken, pause training and return the
                    # previous logged net
                    model = old_net
                    print(
                        'Training finished due to early stopping.\n Actual number of epochs:', epoch)
                    break
                old_valid_loss = valid_loss
                # If the validation has passed, keep the net logged!
                old_net = model
    return model, train_loss_per_epoch, valid_loss_per_epoch


def print_results(net, val_loader, test_loader, train_loss, valid_loss, epochs):
    """Some results needed for evaluation."""

    val_conf = np.zeros((7, 7))
    for xb, yb, lb in val_loader:
        out = net.forward(xb, lb)
        preds = F.softmax(out, dim=-1).argmax(dim=-1)
        acc_val = int((preds == yb).sum()) / len(yb) * 100
    print('Accuracy on validation set:', acc_val, '%')

    # Testing
    predictions = []
    labels = []
    for xb, yb, lb in test_loader:
        # Pass the test data through the network
        out = net.forward(xb, lb)
        # Predict the one with the highest probability
        preds = F.softmax(out, dim=-1).argmax(dim=-1)
        # Need for metrics
        predictions.append(preds.data.numpy())
        labels.append(yb.data.numpy())
        # Get accuracy manually
        acc_test = int((preds == yb).sum()) / len(yb) * 100

    print('Accuracy on test set:', acc_test, '%')

    # Train and validation
    fig = plt.figure(figsize=(10,10))
    plt.ylabel('Train-Validation loss')
    plt.xlabel('Epoch')
    try:
        plt.plot(list(range(1, epochs+1)), train_loss_per_epoch, color='r')
    except:
        print("Train loss can't be printed")
    try:
        # Convert valid losses to printable
        vls = np.array([[l]*5 for l in valid_loss_per_epoch]).flatten()
        plt.plot(list(range(len(vls))), vls, color='b')
    except:
        print("Validation loss can't be printed")
    plt.savefig('plotting/plots/train_valid_loss.png')

    # Print some metrics
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    f1_metric = f1_score(labels, predictions, average='macro')
    print(f'f1 score: {f1_metric}')


