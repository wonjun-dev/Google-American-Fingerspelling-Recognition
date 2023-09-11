## dataset code
import sys

sys.path.append("/sources/dataset/refactor_v3_hdist")

import tensorflow as tf
from data_util import *


def preprocess_fn(
    data,
    max_len=None,
    augment=False,
    use_major=False,
    use_flip_lr=True,
    use_affine=True,
    use_resample=True,
    use_reverse=False,
    rate=0.2,
):
    xy = data["landmark"]  # (-1, NUM_LANDMARK, 2)
    xy = filter_nans_tf(xy)

    # Use Dominate hand only
    if use_major:
        xy = check_major(xy)
    num_lms = tf.shape(xy)[1]

    if augment:
        xy = augment_func(
            xy,
            use_flip_lr=use_flip_lr,
            use_affine=use_affine,
            use_resample=use_resample,
            use_reverse=use_reverse,
            resample_rate=rate,
        )
        if use_reverse:
            data["phrase"] = tf.reverse(data["phrase"], axis=[0])

    # Normalize
    mean = tf_nan_mean(tf.gather(xy, [0], axis=1), axis=[0, 1], keepdims=True)
    mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, xy.dtype), mean)
    std = tf_nan_std(xy, center=mean, axis=[0, 1], keepdims=True)
    xy = (xy - mean) / std

    # Truncation
    if max_len is not None:
        xy = xy[:max_len]
    xy_len = tf.shape(xy)[0]

    # Feature engineering 1: time delta
    dxy = tf.cond(
        tf.shape(xy)[0] > 1,
        lambda: tf.pad(xy[1:] - xy[:-1], [[0, 1], [0, 0], [0, 0]]),
        lambda: tf.zeros_like(xy),
    )
    dxy2 = tf.cond(
        tf.shape(xy)[0] > 1,
        lambda: tf.pad(xy[2:] - xy[:-2], [[0, 2], [0, 0], [0, 0]]),
        lambda: tf.zeros_like(xy),
    )
    # Featue engineering 2: hdist
    mask = tf.linalg.band_part(
        tf.ones((xy_len, 21, 21), dtype=tf.float32), -1, 0
    ) - tf.eye(21, dtype=tf.float32)
    hand = xy[:, -21:, :]
    dist = hand[:, :, tf.newaxis, :] - hand[:, tf.newaxis, :, :]
    dist = tf.sqrt(tf.reduce_sum(dist**2, axis=-1))
    dist = tf.boolean_mask(dist, mask)

    xy = tf.concat(
        [
            tf.reshape(xy, (-1, num_lms * 2)),
            tf.reshape(dxy, (-1, num_lms * 2)),
            tf.reshape(dxy2, (-1, num_lms * 2)),
            tf.reshape(dist, (-1, 210)),
        ],
        axis=-1,
    )

    xy = tf.where(tf.math.is_nan(xy), tf.constant(0.0, xy.dtype), xy)

    return xy, data["phrase"]


def get_tfrec_dataset(
    tfrecs,
    batch_size=64,
    max_len=None,
    max_phrase_len=31,
    augment=False,
    use_major=False,
    use_flip_lr=True,
    use_affine=True,
    use_resample=True,
    use_reverse=False,
    rate=0.2,
    repeat=True,
    shuffle=0,
    drop_remainder=True,
    seed=42,
):
    dataset = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda x: preprocess_fn(
            x,
            max_len=max_len,
            augment=augment,
            use_major=use_major,
            use_flip_lr=use_flip_lr,
            use_affine=use_affine,
            use_resample=use_resample,
            use_reverse=use_reverse,
            rate=rate,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if repeat:
        dataset = dataset.repeat()

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle, seed=seed, reshuffle_each_iteration=True
        )

    if use_major:
        num_lms = NUM_LANDMARK - 21
    else:
        num_lms = NUM_LANDMARK

    padded_shapes = (
        tf.TensorShape([max_len, num_lms * 6 + 210]),
        tf.TensorShape([max_phrase_len]),
    )
    padding_values = (
        tf.constant(-100.0, dtype=tf.float32),
        tf.constant(0, dtype=tf.int32),
    )

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        drop_remainder=drop_remainder,
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    from glob import glob

    tfrecs = glob("/sources/dataset/pid_data_v6/train_tfrecords/record_0.tfrecords")

    tfrec_dataset_dec = get_tfrec_dataset(
        tfrecs, max_len=384, batch_size=2, use_major=True
    )

    for x, y in tfrec_dataset_dec:
        print(x.shape, y.shape)
        break
