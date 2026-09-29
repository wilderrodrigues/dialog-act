import random
import numpy as np

from uu import get_root
from uu.msc.ai.mair.dialog.core.datasets import DatasetFactory
from uu.msc.ai.mair.dialog.model.log_reg_classifier import LogRegModel

random.seed(DatasetFactory.SEED)
np.random.seed(DatasetFactory.SEED)

def test_log_reg_model() -> None:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"
    dataframe = DatasetFactory.load_dataframe(dataset_file, separator=" ")
    targets = DatasetFactory.get_targets_map(dataframe)

    log_reg_model = LogRegModel(dataset_path=dataset_file)
    assert log_reg_model is not None

    acts_val, acts_pred = log_reg_model.train(split=0.5)
    assert acts_val is not None
    assert acts_pred is not None
    assert len(acts_val) == len(acts_pred)

    acts_val_keys = [targets[act] for act in acts_val]
    acts_pred_keys = [targets[act] for act in acts_pred]

    np.allclose(acts_val_keys, acts_pred_keys, rtol=1e-05, atol=1e-08)
