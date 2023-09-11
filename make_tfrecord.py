import os
from glob import glob
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

import tensorflow as tf

parquet_path = "/sources/dataset/pid_data_v6/supplemental_landmarks"
meta_data = pd.read_csv("/sources/dataset/pid_data_v6/supplemental_metadata.csv")
pids = meta_data.participant_id.unique()
pids.sort()
parquet_list = [os.path.join(parquet_path, f"{pid}.parquet") for pid in pids]
with open(
    "/sources/dataset/pid_data_v6/character_to_prediction_index_blank.json", "r"
) as f:
    char_to_idx_dict = json.load(f)

# with open(
#     "/sources/dataset/pid_data_v4_decoder/character_to_prediction_index_decoder.json",
#     "r",
# ) as f:
#     char_to_idx_dict_decoder = json.load(f)


def make_single_example(data, phrase, phrase_dec):
    feature = {
        # "landmark": tf.train.Feature(
        #     float_list=tf.train.FloatList(value=data.flatten())
        # ),
        # "phrase": tf.train.Feature(
        #     bytes_list=tf.train.BytesList(value=[phrase.encode("utf-8")])
        # ),
        "landmark": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data.tobytes()])
        ),
        "phrase": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[phrase.tobytes()])
        ),
        # "phrase_dec": tf.train.Feature(
        #     bytes_list=tf.train.BytesList(value=[phrase_dec.tobytes()])
        # ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def char_to_idx(phrase, is_decoder=False):
    if not is_decoder:
        arr = np.array([char_to_idx_dict[ph] for ph in phrase], dtype=np.int32)
    # else:
    #     arr = np.array([char_to_idx_dict_decoder[ph] for ph in phrase], dtype=np.int32)
    return arr


def add_special_tokens(phrase):
    ph = "<" + phrase + ">"
    return ph


for idx, parq_path in enumerate(parquet_list):
    pid = parq_path.split("/")[-1].split(".")[0]
    parq = pd.read_parquet(parq_path)
    seq_ids = parq.index.unique()
    with tf.io.TFRecordWriter(
        f"/sources/dataset/pid_data_v6/supplemental_tfrecords/record_{pid}.tfrecords"
    ) as writer:
        for seq_id in seq_ids:
            ph = meta_data.loc[meta_data["sequence_id"] == seq_id]

            ph = ph["phrase"].to_string(index=False)
            # ph_dec = add_special_tokens(ph)

            ph = char_to_idx(ph, is_decoder=False)  # convert phrase to list of indices
            # ph_dec = char_to_idx(
            #     ph_dec, is_decoder=True
            # )  # convert phrase to list of indices
            seq_data = parq.loc[seq_id]
            # seq_data = seq_data.sort_values(by="frame") # v1,v2,v3
            seq_data = np.array(seq_data, dtype=np.float32)
            # seq_data = np.nan_to_num(seq_data, nan=0.0)  # replace nan with 0.0
            # example = make_single_example(seq_data[:, 1:], ph)  # drop frame column
            example = make_single_example(seq_data, ph, None)
            # example = make_single_example(seq_data, ph, ph_dec)
            serialized = example.SerializeToString()
            writer.write(serialized)
