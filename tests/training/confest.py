import pytest

from training.make_sample_dataset import make_dataset
from training.split_dataset import split_dataset


@pytest.fixture(scope="session")
def raw_sample(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw")
    make_dataset(out, groups_per_class=30, images_per_group=3, size=64, seed=0)
    return out


@pytest.fixture(scope="session")
def split_data(tmp_path_factory, raw_sample):
    out = tmp_path_factory.mktemp("split") / "dataset"
    split_dataset(raw_sample, out, group_by="prefix", group_regex=r"^(.*?)__", seed=1)
    return out