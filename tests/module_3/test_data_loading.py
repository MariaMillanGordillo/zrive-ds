import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.module_3.data_loading import download_s3_folder, load_data


@patch("src.module_3.data_loading.boto3.client")
def test_download_s3_folder_downloads_files(mock_boto, tmp_path):
    # Mock S3 client
    mock_s3 = Mock()
    mock_boto.return_value = mock_s3

    # Response with two files
    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "groceries/box_builder_dataset/file1.csv"},
            {"Key": "groceries/box_builder_dataset/file2.csv"}
        ]
    }

    download_s3_folder("my-bucket", "groceries/box_builder_dataset/", tmp_path)

    assert mock_s3.download_file.call_count == 2


@patch("src.module_3.data_loading.download_s3_folder")
def test_load_data_downloads_if_missing(mock_download, tmp_path):
    local_file = tmp_path / "data.csv"
    mock_download.return_value = None

    df_expected = pd.DataFrame({"a": [1, 2, 3]})
    df_expected.to_csv(local_file, index=False)

    df = load_data(local_file, download_if_missing=True)
    pd.testing.assert_frame_equal(df, df_expected)


def test_load_data_raises_if_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nonexistent.csv", download_if_missing=False)
