### data util (tfrecords reading, parsing, etc.)
import json
from glob import glob

import tensorflow as tf
import numpy as np

with open("/sources/dataset/pid_data/cols_to_idx.json", "r") as f:
    cols_to_idx = json.load(f)
NORM_REF = ["x_face_5", "y_face_5"]
LIP = [
    f"{j}_face_{i}"
    for i in [
        0,
        61,
        185,
        40,
        39,
        37,
        267,
        269,
        270,
        409,
        291,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        95,
        88,
        178,
        87,
        14,
        317,
        402,
        318,
        324,
        308,
    ]
    for j in ["x", "y"]
]
LHAND = [f"{j}_left_hand_{i}" for i in range(21) for j in ["x", "y"]]
RHAND = [f"{j}_right_hand_{i}" for i in range(21) for j in ["x", "y"]]
USE_COLS = NORM_REF + LIP + LHAND + RHAND
USE_COLS = [cols_to_idx[col] for col in USE_COLS]
NUM_LANDMARK = int(len(USE_COLS) // 2)

####### dataset length #######
TRAIN_TRAIN_LEN = 51488
TRAIN_VAL_LEN = 12569
SUPP_TRAIN_LEN = 39404
SUPP_VAL_LEN = 10775


def prepare_data(data="train", val_fold=1, version="v4", supptrain=False):
    train_pids = []
    val_pids = []

    if version == "v1":
        data_ver = "pid_data"
        print("Data version: v1")
    elif version == "v2":
        data_ver = "pid_data_v2"
        print("Data version: v2")
    elif version == "v3":
        data_ver = "pid_data_v3"
        print("Data version: v3")
    elif version == "v4":
        data_ver = "pid_data_v4"
        print("Data version: v4")
    elif version == "v4_decoder":
        data_ver = "pid_data_v4_decoder"
        print("Data version: v4_decoder")
    elif version == "v5":
        data_ver = "pid_data_v5"
        print("Data version: v5")
    elif version == "v6":
        data_ver = "pid_data_v6"
        print("Data version: v6")

    if val_fold != -1:
        for i in range(1, 6):
            if i == val_fold:
                val_pids = np.load(
                    f"/sources/dataset/{data_ver}/folds/{data}_fold_{i}.npy"
                ).tolist()
            else:
                train_pids.extend(
                    np.load(
                        f"/sources/dataset/{data_ver}/folds/{data}_fold_{i}.npy"
                    ).tolist()
                )

        tfrecs = glob(
            f"/sources/dataset/{data_ver}/{data}_tfrecords/record_*.tfrecords"
        )

        train_tfrecs = [
            tfrec
            for tfrec in tfrecs
            if int(tfrec.split("_")[-1].split(".")[0]) in train_pids
        ]
        val_tfrecs = [
            tfrec
            for tfrec in tfrecs
            if int(tfrec.split("_")[-1].split(".")[0]) in val_pids
        ]

        return train_tfrecs, val_tfrecs

    elif val_fold == -1:
        train_tfrecs = glob(
            f"/sources/dataset/{data_ver}/{data}_tfrecords/record_*.tfrecords"
        )
        if supptrain:
            supp_tfrecs = glob(
                f"/sources/dataset/{data_ver}/supplemental_tfrecords/record_*.tfrecords"
            )
            train_tfrecs.extend(supp_tfrecs)
        return train_tfrecs, None


def tf_nan_mean(x, axis=[0, 1], keepdims=True):
    return tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims
    ) / tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)),
        axis=axis,
        keepdims=keepdims,
    )


def tf_nan_std(x, center=None, axis=[0, 1], keepdims=True):
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))


def filter_nans_tf(x):
    # 모든 키포인트가 nan인 경우 해당 frame 제거
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(x), axis=[-2, -1]))
    x = tf.boolean_mask(x, mask, axis=0)
    return x


def parse_tfrecord_fn(dataset):
    feature_description = {
        "landmark": tf.io.FixedLenFeature([], tf.string),
        "phrase": tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(dataset, feature_description)
    landmark = tf.reshape(
        tf.io.decode_raw(features["landmark"], tf.float32), (-1, len(USE_COLS))
    )
    # landmark = tf.gather(landmark, use_cols, axis=1)
    landmark = tf.reshape(landmark, (-1, NUM_LANDMARK, 2))
    phrase = tf.io.decode_raw(features["phrase"], tf.int32)

    out = {}
    out["landmark"] = landmark
    out["phrase"] = phrase

    return out


def check_major(x):
    lhand_power = tf.reduce_sum(
        tf.where(
            tf.math.is_nan(x[:, 41:62]),
            tf.zeros_like(x[:, 41:62]),
            tf.ones_like(x[:, 41:62]),
        )
    )
    rhand_power = tf.reduce_sum(
        tf.where(
            tf.math.is_nan(x[:, 62:]),
            tf.zeros_like(x[:, 62:]),
            tf.ones_like(x[:, 62:]),
        )
    )

    if lhand_power > rhand_power:
        x = x[:, :62]
    else:
        x = tf.concat([x[:, :41], x[:, 62:]], axis=1)
    return x


def augment_func(
    x,
    use_resample=True,
    use_flip_lr=True,
    use_affine=True,
    use_reverse=False,
    resample_rate=0.2,
):
    if use_resample and tf.random.uniform(()) < 0.8:
        x = resample(x, rate=(1.0 - resample_rate, 1.0 + resample_rate))
    if use_flip_lr and tf.random.uniform(()) < 0.5:
        x = flip_lr(x)
    if use_affine and tf.random.uniform(()) < 0.75:
        x = affine_transform(x)
    if use_reverse and tf.random.uniform(()) < 0.5:
        x = temporal_reverse(x)

    return x


def flip_lr(x):
    x, y = tf.unstack(x, axis=-1)
    x = 1 - x
    new_x = tf.stack([x, y], axis=-1)

    LLIP = [15, 14, 13, 12, 2, 3, 4, 5, 6, 34, 33, 32, 31, 21, 22, 23, 24, 25]  # index
    RLIP = [17, 18, 19, 20, 11, 10, 9, 8, 7, 36, 37, 38, 39, 40, 30, 29, 28, 27]

    new_x = tf.transpose(new_x, [1, 0, 2])
    llip = tf.gather(new_x, LLIP, axis=0)
    rlip = tf.gather(new_x, RLIP, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LLIP)[..., None], rlip)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RLIP)[..., None], llip)
    new_x = tf.transpose(new_x, [1, 0, 2])
    return new_x


def affine_transform(
    x, scale=(0.8, 1.2), shear=(-0.15, 0.15), shift=(-0.1, 0.1), degree=(-30, 30)
):
    center = tf.constant([0.5, 0.5])
    if scale is not None:
        scale = tf.random.uniform((), *scale)
        x = x * scale

    if shear is not None:
        shear_x = shear_y = tf.random.uniform((), *shear)
        if tf.random.uniform(()) < 0.5:
            shear_x = 0.0
        else:
            shear_y = 0.0
        shear_mat = tf.identity([[1.0, shear_x], [shear_y, 1.0]])
        x = x @ shear_mat
        center = center + [shear_y, shear_x]

    if degree is not None:
        x -= center
        degree = tf.random.uniform((), *degree)
        radian = degree / 180 * np.pi
        c = tf.math.cos(radian)
        s = tf.math.sin(radian)
        rotate_mat = tf.identity(
            [
                [c, s],
                [-s, c],
            ]
        )
        x = x @ rotate_mat
        x = x + center

    if shift is not None:
        shift = tf.random.uniform((), *shift)
        x = x + shift

    return x


def interp1d_(x, target_len, method="random"):
    target_len = tf.maximum(1, target_len)
    if method == "random":
        if tf.random.uniform(()) < 0.33:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), "bilinear")
        else:
            if tf.random.uniform(()) < 0.5:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), "bicubic")
            else:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), "nearest")
    else:
        x = tf.image.resize(x, (target_len, tf.shape(x)[1]), method)
    return x


def resample(x, rate=(0.8, 1.2)):
    rate = tf.random.uniform((), rate[0], rate[1])
    length = tf.shape(x)[0]
    new_size = tf.cast(tf.cast(length, tf.float32) * rate, tf.int32)
    new_x = interp1d_(x, new_size)
    return new_x


def temporal_reverse(
    x,
):
    x = tf.reverse(x, axis=[0])
    return x
