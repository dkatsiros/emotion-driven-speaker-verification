"""Export diagrams."""
import os
import matplotlib.pyplot as plt
from utils.emodb import idx2emotion
from config import PLOTS_FOLDER


def class_statistics(categories=None, save=True, filename='class_stats.png'):
    """Export class percentages."""
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
    plt.figure(figsize=(10,10))
    plt.bar(x_axis, y_axis)
    # Labels
    plt.xlabel('Emotion')
    plt.ylabel('Percentage in total wavs')
    plt.title('Classes')
    # Export diagram
    plt.savefig(file)
