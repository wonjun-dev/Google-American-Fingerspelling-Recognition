### 공통 모델 모듈들
import tensorflow as tf
import numpy as np


class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(
        self, kernel_size=9, dilation_rate=1, use_bias=False, name="", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D(
            (dilation_rate * (kernel_size - 1), 0), name=name + "_causal_pad"
        )
        self.dw_conv1d = tf.keras.layers.DepthwiseConv1D(
            kernel_size,
            strides=1,
            dilation_rate=dilation_rate,
            padding="valid",
            use_bias=use_bias,
            name=name + "_dw_conv1d",
        )
        self.supports_masking = True

    def call(self, x):
        x = self.causal_pad(x)
        x = self.dw_conv1d(x)
        return x


# ECBAM_SWISH
class ECBAM(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, name="ecbam", **kwargs):
        super().__init__(**kwargs)
        self.channel_att_conv = tf.keras.layers.Conv1D(
            1, kernel_size, strides=1, padding="same", use_bias=False
        )
        self.causal_pad = tf.keras.layers.ZeroPadding1D(
            ((kernel_size - 1), 0), name=name + "_causal_pad"
        )
        self.temporal_att_conv = tf.keras.layers.Conv1D(
            filters=1,
            strides=1,
            kernel_size=kernel_size,
            padding="valid",
            use_bias=True,
            name=name + "_temporal_att_conv",
        )
        self.supports_masking = True

    def call(self, x, mask=None):
        ### channel attention
        gap = tf.keras.layers.GlobalAveragePooling1D()(x, mask=mask)

        # global max pooling doesn't support mask argument.
        if mask is not None:
            gmp_mask = ~mask
            gmp_mask = tf.cast(gmp_mask, x.dtype) * -1e9
            gmp_mask = tf.expand_dims(gmp_mask, axis=-1)
            gmp = tf.keras.layers.GlobalMaxPooling1D()(x + gmp_mask)
        else:
            gmp = tf.keras.layers.GlobalMaxPooling1D()(x)

        ## channel att mlp
        channel_att = tf.expand_dims(gap + gmp, axis=-1)
        channel_att = self.channel_att_conv(channel_att)
        channel_att = tf.squeeze(channel_att, axis=-1)
        channel_att = tf.math.sigmoid(channel_att)

        # apply attention
        channel_att = tf.multiply(x, tf.expand_dims(channel_att, axis=1))

        ### temporal attention
        t_gap = tf.reduce_mean(channel_att, axis=-1, keepdims=True)
        t_gmp = tf.reduce_max(channel_att, axis=-1, keepdims=True)
        temporal_att = tf.concat([t_gap, t_gmp], axis=-1)
        ## temporal att conv
        temporal_att = self.causal_pad(temporal_att)
        temporal_att = self.temporal_att_conv(temporal_att)
        temporal_att = tf.math.sigmoid(temporal_att)

        ## apply attention
        temporal_att = tf.multiply(channel_att, temporal_att)

        # residual connection
        # return tf.keras.layers.ReLU()(x + temporal_att)
        return tf.keras.layers.Activation("swish")(x + temporal_att)


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=4, dropout=0.2, lookahead=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim**-0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.lookahead = lookahead
        self.supports_masking = True

    def call(self, x, mask=None):
        qkv = self.qkv(x)
        qkv = tf.keras.layers.Permute((2, 1, 3))(
            tf.keras.layers.Reshape(
                (-1, self.num_heads, self.dim * 3 // self.num_heads)
            )(qkv)
        )
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        if mask is not None and self.lookahead:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.dropout_1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(
            tf.keras.layers.Permute((2, 1, 3))(x)
        )
        x = self.proj(x)
        return x


class LateDropout(tf.keras.layers.Layer):
    def __init__(
        self, rate, noise_shape=None, start_step=0, name="late_dropout", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(
            0, dtype="int64", aggregation=agg, trainable=False
        )

    def call(self, inputs, training=False):
        x = tf.cond(
            self._train_counter < self.start_step,
            lambda: inputs,
            lambda: self.dropout(inputs, training=training),
        )
        if training:
            self._train_counter.assign_add(1)
        return x


class LateDropoutV2(tf.keras.layers.Layer):
    def __init__(self, init_rate, start_step, final_step, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.supports_masking = True
        self.rate = init_rate
        self.start_step = start_step
        self.final_step = final_step

        self.dropout1 = tf.keras.layers.Dropout(init_rate)
        self.dropout2 = tf.keras.layers.Dropout(init_rate * 2)
        self.dropout3 = tf.keras.layers.Dropout(init_rate * 3)

    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(
            0, dtype="int64", aggregation=agg, trainable=False
        )

    def call(self, inputs, training=False):
        x = tf.cond(
            self._train_counter < self.start_step,
            lambda: self.dropout1(inputs, training=training),
            lambda: tf.cond(
                self._train_counter < self.final_step,
                lambda: self.dropout2(inputs, training=training),
                lambda: self.dropout3(inputs, training=training),
            ),
        )
        if training:
            self._train_counter.assign_add(1)
        return x


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int = 192, name=None):
        super(PositionalEncodingLayer, self).__init__(name=name)

        self.d_model = d_model
        self.supports_masking = True

    def call(self, x, mask=None):
        max_len = x.shape[1]
        position = tf.range(0, max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, self.d_model, 2, dtype=tf.float32)
            * -(np.log(10000.0) / self.d_model)
        )

        # Compute the positional encodings for even indices
        pe_even = tf.sin(position * div_term)
        # Compute the positional encodings for odd indices
        pe_odd = tf.cos(position * div_term)

        # Generate a tensor of shape [max_len, d_model]
        # by interleaving the tensors pe_even and pe_odd
        pe = tf.reshape(tf.stack([pe_even, pe_odd], axis=-1), [-1, self.d_model])

        pe = tf.expand_dims(pe, 0)
        x = x + pe[:, :max_len]
        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self, hidden_dim, kernel_size, cbam_kernel_size, expand_ratio, dropout
    ):
        super(ConvBlock, self).__init__()
        self.hidden_din = hidden_dim
        self.kernel_size = kernel_size
        self.cbam_kernel_size = cbam_kernel_size
        self.dropout = dropout
        self.supports_masking = True

        # self.ln = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(
            hidden_dim * expand_ratio, use_bias=True, activation="swish"
        )
        self.conv = CausalDWConv1D(kernel_size, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.95)
        self.cbam = ECBAM(kernel_size=cbam_kernel_size)
        self.dense2 = tf.keras.layers.Dense(hidden_dim, use_bias=True)
        self.dropout = tf.keras.layers.Dropout(dropout, noise_shape=(None, 1, 1))

    def call(self, x, mask=None):
        skip = x
        # x = self.ln(x)
        x = self.dense1(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.cbam(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = x + skip
        return x


class SABlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads, expand_ratio, dropout):
        super(SABlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.supports_masking = True

        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.95)
        # self.ln = tf.keras.layers.LayerNormalization()
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.dropout1 = tf.keras.layers.Dropout(dropout, noise_shape=(None, 1, 1))
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.95)
        self.dense1 = tf.keras.layers.Dense(
            hidden_dim * expand_ratio, use_bias=False, activation="swish"
        )
        self.dense2 = tf.keras.layers.Dense(hidden_dim, use_bias=False)
        self.dropout2 = tf.keras.layers.Dropout(dropout, noise_shape=(None, 1, 1))

    def call(self, x, mask=None):
        skip = x
        x = self.bn1(x)
        # x = self.ln(x)
        x = self.attention(x, mask=mask)
        x = self.dropout1(x)
        x = x + skip

        skip = x
        x = self.bn2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = x + skip

        return x


# Conv, Transformer, BiLSTM
class EncoderLayerTC(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        kernel_size,
        cbam_kernel_size,
        num_heads,
        expand_ratio,
        cls_expand_ratio,
        dropout,
        name=None,
    ):
        super(EncoderLayerTC, self).__init__(name=name)
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.cbam_kernel_size = cbam_kernel_size
        self.dropout = dropout
        self.supports_masking = True

        self.conv_block1 = ConvBlock(
            hidden_dim, kernel_size, cbam_kernel_size, expand_ratio, dropout
        )
        self.conv_block2 = ConvBlock(
            hidden_dim, kernel_size, cbam_kernel_size, expand_ratio, dropout
        )
        self.conv_block3 = ConvBlock(
            hidden_dim, kernel_size, cbam_kernel_size, expand_ratio, dropout
        )
        self.attn_block1 = SABlock(hidden_dim, num_heads, expand_ratio, dropout)
        self.attn_block2 = SABlock(hidden_dim, num_heads, expand_ratio, dropout)
        self.attn_block3 = SABlock(hidden_dim, num_heads, expand_ratio, dropout)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.ln3 = tf.keras.layers.LayerNormalization()
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                int(hidden_dim // 2),
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
            )
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(
            hidden_dim * cls_expand_ratio, use_bias=True, activation="swish"
        )

    def call(self, x, mask=None):
        x = self.attn_block1(x)
        x = self.conv_block1(x)
        x = self.ln1(x)

        x = self.attn_block2(x)
        x = self.conv_block2(x)
        x = self.ln2(x)

        x = self.attn_block3(x)
        x = self.conv_block3(x)
        x = self.ln3(x)

        x = self.lstm(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x
