from pathlib import Path

import matplotlib.pyplot as plt
import numpy.typing as npt
import logging
from sklearn.metrics import ConfusionMatrixDisplay


logging.basicConfig(level=logging.INFO)

def plot_confusion_matrix(confusion_matrix: npt.NDArray | None, targets_map: dict[str, int], output_dir: Path) -> None:
    if confusion_matrix is not None:
        logging.info("Plotting Confusion Matrix...")
        display_labels = [target for target, _ in sorted(targets_map.items(), key=lambda item: item[1])]
        figure_size = max(8, len(display_labels) * .75)
        figure, axes = plt.subplots(figsize=(figure_size, figure_size), constrained_layout=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical", values_format="d", ax=axes)

        axes.set_title("Confusion Matrix")
        figure.savefig(output_dir, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(figure)
