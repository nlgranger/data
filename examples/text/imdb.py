# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper

from .utils import _add_docstring_header, _create_dataset_directory, _wrap_split_argument

URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

MD5 = "7c2ac02c03563afcf9b574c7e56c153a"

NUM_LINES = {
    "train": 25000,
    "test": 25000,
}

_PATH = "aclImdb_v1.tar.gz"

DATASET_NAME = "IMDB"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def IMDB(root, split):
    """Demonstrates complex use case where each sample is stored in separate file and compressed in tar file
    Here we show some fancy filtering and mapping operations.
    Filtering is needed to know which files belong to train/test and neg/pos label
    Mapping is needed to yield proper data samples by extracting label from file name
        and reading data from file
    """

    url_dp = IterableWrapper([URL])
    # cache data on-disk
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, os.path.basename(x)),
        hash_dict={os.path.join(root, os.path.basename(URL)): MD5},
        hash_type="md5",
    )
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_dp = FileOpener(cache_dp, mode="b")

    # stack TAR extractor on top of load files data pipe
    extracted_files = cache_dp.load_from_tar()

    # filter the files as applicable to create dataset for given split (train or test)
    filter_files = extracted_files.filter(
        lambda x: Path(x[0]).parts[-3] == split and Path(x[0]).parts[-2] in ["pos", "neg"]
    )

    # map the file to yield proper data samples
    return filter_files.map(lambda x: (Path(x[0]).parts[-2], x[1].read().decode("utf-8")))
