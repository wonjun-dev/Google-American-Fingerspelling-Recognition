## build model functions
import sys

sys.path.append("/sources/dataset/refactor_v3_hdist")

from model_util import *
from data_util import NUM_LANDMARK

PAD_VALUE = -100.0


def get_model(
    max_len=384,
    hidden_dim=256,
    kernel_size=11,
    cbam_kernel_size=11,
    num_heads=4,
    expand_ratio=4,
    inter_cls_expand_ratio=1,
    head_cls_expand_ratio=4,
    dropout=0.2,
    late_dropout=0.5,
    late_dropout_start=25,
    use_major=True,
    class_num=60,
):
    if use_major:
        num_lmp_ch = (NUM_LANDMARK - 21) * 6 + 210
    else:
        num_lmp_ch = NUM_LANDMARK * 6 + 210

    inp_enc = tf.keras.layers.Input(shape=(max_len, num_lmp_ch), name="input_encoder")
    inp_enc_masking = tf.keras.layers.Masking(
        mask_value=PAD_VALUE,
        input_shape=(max_len, num_lmp_ch),
        name="input_encoder_mask",
    )
    late_dropout = LateDropout(late_dropout, start_step=late_dropout_start)

    x = inp_enc_masking(inp_enc)
    x = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="stem_dense")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name="stem_bn")(x)

    x = PositionalEncodingLayer(d_model=hidden_dim, name="positional_encoding_layer_1")(
        x
    )

    x = EncoderLayerTC(
        hidden_dim,
        kernel_size,
        cbam_kernel_size,
        num_heads,
        expand_ratio,
        inter_cls_expand_ratio,
        dropout,
        name=f"encoder_layer_1",
    )(x)
    x = tf.keras.layers.LayerNormalization(name="inter_ln")(x)
    inter_x1 = tf.keras.layers.Dense(
        class_num, use_bias=False, activation=None, name="inter_cls"
    )(x)
    inter_x_prob = tf.keras.activations.softmax(inter_x1, axis=-1)
    inter_x_prob = tf.keras.layers.Dense(
        hidden_dim, use_bias=False, name="inter_dense"
    )(inter_x_prob)
    x = tf.keras.layers.Add(name="add_1")([x, inter_x_prob])

    x = EncoderLayerTC(
        hidden_dim,
        kernel_size,
        cbam_kernel_size,
        num_heads,
        expand_ratio,
        head_cls_expand_ratio,
        dropout,
        name=f"encoder_layer_2",
    )(x)
    x = tf.keras.layers.LayerNormalization(name="head_ln")(x)
    x = late_dropout(x)
    x = tf.keras.layers.Dense(
        class_num, use_bias=False, activation=None, name="head_cls"
    )(x)

    return tf.keras.Model(
        inputs=inp_enc,
        outputs=[x, inter_x1],
    )


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[5], "GPU")
    model = get_model(
        max_len=384,
        hidden_dim=192,
        late_dropout=0.5,
        late_dropout_start=5000,
        use_major=False,
        inter_ctc=False,
    )
    model.summary()
