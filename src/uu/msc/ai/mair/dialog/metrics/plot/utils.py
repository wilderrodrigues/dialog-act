from pathlib import Path

import matplotlib.pyplot as plt
import numpy.typing as npt
import logging
from sklearn.metrics import ConfusionMatrixDisplay


logging.basicConfig(level=logging.INFO)

def plot_confusion_matrix(confusion_matrix: npt.NDArray | None, targets_map: dict[str, int], output_dir: Path):
    if confusion_matrix is not None:
        logging.info("Plotting Confusion Matrix...")
        # TODO [Wilder]: Since we are using only one fold when in the grouped strategy, one target is missing and hence
        #   causing problems with the confusion matrix. We will fix that before the final release of our code.
        if confusion_matrix.shape[0] == 13:
            display_labels = list(targets_map.keys())[:-1]
        else:
            display_labels = list(targets_map.keys())
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")

        plt.title("Confusion Matrix")
        plt.savefig(output_dir / "nn_confusion_matrix.png")
        plt.show()
