"""Export diagrams."""
import os
import matplotlib.pyplot as plt
from core.config import PLOTS_FOLDER
import numpy as np


def class_statistics(categories=None, save=True, filename='class_stats.png'):
    """Export class percentages."""
    from utils.emodb import idx2emotion
    # Empty stats
    x_axis = []
    y_axis = []
    # Get everything
    for emotion_idx, li in categories.items():
        y_axis.append(len(li))
        x_axis.append(idx2emotion(emotion_idx))
    # Normalize to 100
    total_wavs = sum(y_axis)
    y_axis = [y/total_wavs * 100 for y in y_axis]
    if save is False:
        print(f'Total wavs imported: {total_wavs}')
        print(f'Percentages of different emotion classes:')
        # Just print stats
        for emotion, percent in zip(x_axis, y_axis):
            percent_less_decimals = str(percent)[:5]
            print(f'{emotion} : {percent_less_decimals}%')
        # Exit
        return True
    # Save figure
    file = os.path.join(PLOTS_FOLDER, filename)
    # Set dimensions
    plt.figure(figsize=(10, 10))
    plt.bar(x_axis, y_axis)
    # Labels
    plt.xlabel('Emotion')
    plt.ylabel('Percentage in total wavs')
    plt.title('Classes')
    # Export diagram
    plt.savefig(file)


def dataloader_stats(dataloader, filename='dataloader_statistics.png', dataset='EMODB'):
    """Get a dataloader and check percentage of each class."""
    from utils.emodb import idx2emotion
    # Empty counter
    cnt = {i: 0 for i in range(7)}
    for batch in dataloader:
        # Get label tensor
        _, label, _ = batch
        # Add
        for l in label.numpy():
            cnt[int(l)] += 1

    x_axis = []
    y_axis = []

    for label, value in cnt.items():
        if dataset == "EMODB":
            x_axis.append(idx2emotion(label))
        elif dataset == "IEMOCAP":
            x_axis.append(label)
        y_axis.append(value)

    plt.figure(figsize=(10, 10))
    plt.bar(x_axis, y_axis)
    plt.xlabel('Classes')
    plt.ylabel('Samples')
    plt.title('Dataloader Sample Analysis')

    # Save figure
    file = os.path.join(PLOTS_FOLDER, filename)
    plt.savefig(file)


def samples_lengths(dataloaders=[], dataset_name='emodb'):
    """Create a diagram with len of time series for the dataset."""
    total_samples = sum([len(dtld.dataset) for dtld in dataloaders])
    lengths = np.empty(total_samples, dtype=int)
    idx = 0
    for dataloader in dataloaders:
        for _, _, length_tensor in dataloader:
            lengths[idx:idx+len(length_tensor)
                    ] = length_tensor.to('cpu').numpy()
            idx += len(length_tensor)
    plt.figure(figsize=(20, 10))
    step = 10
    lengths = np.sort(lengths, kind='heapsort')
    print('compute')
    bins = list(range(0, (np.max(lengths)//step + 1) * step, step))
    plt.hist(x=lengths, bins=bins, edgecolor='k')
    print('hist done')
    plt.xticks(bins)
    plt.savefig(f'./{dataset_name}_lengths_distribution.png')


def plot_iemocap_classes_population(categories=None, save=True, filename="iemocap.png"):
    """Export class percentages."""
    from utils.iemocap import idx2emotion
    # Empty stats
    x_axis = []
    y_axis = []
    # Get everything
    for emotion_idx, li in categories.items():
        y_axis.append(li)
        x_axis.append(idx2emotion(emotion_idx))
    # Normalize to 100
    total_wavs = sum(y_axis)
    y_axis = [y/total_wavs * 100 for y in y_axis]
    if save is False:
        print(f'Total wavs imported: {total_wavs}')
        print(f'Percentages of different emotion classes:')
        # Just print stats
        for emotion, percent in zip(x_axis, y_axis):
            percent_less_decimals = str(percent)[:5]
            print(f'{emotion} : {percent_less_decimals}%')
        # Exit
        return True
    # Save figure
    file = os.path.join(PLOTS_FOLDER, filename)
    # Set dimensions
    plt.figure(figsize=(10, 10))
    plt.bar(x_axis, y_axis)
    # Labels
    plt.xlabel('Emotion')
    plt.ylabel('Percentage in total wavs')
    plt.title('Classes')
    # Export diagram
    plt.savefig(file)
