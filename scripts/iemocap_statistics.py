# import sys
from plotting.class_stats import plot_iemocap_classes_population
from utils.load_dataset import load_IEMOCAP
from utils.iemocap import get_categories_population_dictionary
# sys.path.append('../plotting')


if __name__ == "__main__":
    # Get all sample labels
    _, y_train, _, y_test, _, y_val, = load_IEMOCAP()
    # Set the number of classes
    NUM_CLASSES = 9
    # Get category --to-> indexes mapping
    cat = get_categories_population_dictionary(
        y_train + y_test + y_val, n_classes=NUM_CLASSES)
    # Make plot
    plot_iemocap_classes_population(categories=cat,
                                    save=True,
                                    filename=f'iemocap_classes_population_{NUM_CLASSES}cl.png')
