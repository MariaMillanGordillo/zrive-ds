import pytest
from unittest.mock import Mock, patch
from src.module_3.data_loading import (
    download_s3_folder,
    load_data,
)
from src.module_3.preprocessing import (
    filter_orders,
    temporal_split_by_order,
)
from src.module_3.train_model import (
    train_logistic_regression,
    plot_confusion_matrix,
    plot_roc_pr,
)