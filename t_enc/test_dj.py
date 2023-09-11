import time
import os

import tensorflow as tf

from dataloader import prepare_data, get_tfrec_dataset
from config import parser
args = parser.parse_args()

gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[args.gpu], "GPU")


def main():
    train_tfrecs, val_tfrecs = prepare_data(data="train", val_fold=1)
    train_dataset = get_tfrec_dataset(
        train_tfrecs, batch_size=32, max_len=384, repeat=False
    )
    val_dataset = get_tfrec_dataset(
        val_tfrecs, batch_size=32, max_len=384, repeat=False
    )


if __name__ == "__main__":
    main()
    time.sleep(86400 * 3)
