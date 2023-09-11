### 공통 loss 함수
import tensorflow as tf


@tf.function
def ctc_loss(
    y_true,
    y_pred,
    input_length,
    label_length,
    blank=0,
    name="ctc_loss",
):
    return tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logit_length=tf.cast(input_length, tf.int32),
        logits=tf.cast(y_pred, tf.float32),
        label_length=tf.cast(label_length, tf.int32),
        logits_time_major=False,
        blank_index=blank,
        name=name,
    )


@tf.function
def focal_ctc_loss(
    y_true,
    y_pred,
    input_length,
    label_length,
    blank=0,
    alpha=0.5,
    gamma=2,
    name="ctc_loss",
):
    ctc_loss = tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logit_length=tf.cast(input_length, tf.int32),
        logits=tf.cast(y_pred, tf.float32),
        label_length=tf.cast(label_length, tf.int32),
        logits_time_major=False,
        blank_index=blank,
        name=name,
    )
    p = tf.exp(-ctc_loss) + tf.keras.backend.epsilon()
    focal_ctc_loss = tf.multiply(tf.multiply(alpha, tf.pow((1 - p), gamma)), ctc_loss)
    return focal_ctc_loss


@tf.function
def custom_categorical_crossentropy(
    y_true,
    y_pred,
    label_smoothing=0.0,
    ignore_class=0,
    from_dense=True,
    from_logits=True,
):
    num_classes = tf.shape(y_pred)[-1]

    # mask for ignore class
    mask = tf.math.logical_not(tf.math.equal(y_true, ignore_class))  # [B, T]
    mask = tf.cast(mask, dtype=y_pred.dtype)

    if from_dense:
        y_true = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_pred = tf.clip_by_value(
        y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()
    )  # [B, T, C]

    # label smoothing
    y_true = y_true * (1.0 - label_smoothing) + (
        label_smoothing / tf.cast(num_classes, tf.float32)
    )  # [B, T, C]
    loss = tf.reduce_sum(tf.multiply(y_true, -tf.math.log(y_pred)), axis=-1)  # [B, T]
    loss = tf.multiply(loss, mask)
    loss = tf.reduce_sum(loss, axis=-1) / tf.reduce_sum(mask, axis=-1)  # [B]
    return loss


@tf.function
def focal_categorical_crossentropy(
    y_true,
    y_pred,
    gamma=2.0,
    alpha=0.25,
    label_smoothing=0.0,
    ignore_class=0,
    from_dense=True,
    from_logits=True,
):
    num_classes = tf.shape(y_pred)[-1]

    # mask for ignore class
    mask = tf.math.logical_not(tf.math.equal(y_true, ignore_class))  # [B, T]
    mask = tf.cast(mask, dtype=y_pred.dtype)

    if from_dense:
        y_true = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_pred = tf.clip_by_value(
        y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()
    )  # [B, T, C]

    # label smoothing
    y_true = y_true * (1.0 - label_smoothing) + (
        label_smoothing / tf.cast(num_classes, tf.float32)
    )  # [B, T, C]
    loss = -1 * y_true * (alpha * tf.pow((1 - y_pred), gamma) * tf.math.log(y_pred))
    loss = tf.reduce_sum(loss, axis=-1)  # [B, T]
    loss = tf.multiply(loss, mask)
    loss = tf.reduce_sum(loss, axis=-1) / tf.reduce_sum(mask, axis=-1)  # [B]
    return loss


class CTCLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        blank=0,
        name=None,
        focal=False,
    ):
        super(CTCLoss, self).__init__(
            reduction=tf.keras.losses.Reduction.NONE, name=name
        )
        self.blank = blank
        self.focal = focal

    def call(
        self,
        y_true,
        y_pred,
    ):
        if self.focal:
            loss = focal_ctc_loss(
                y_pred=y_pred["logits"],
                input_length=y_pred["logits_length"],
                y_true=tf.sparse.from_dense(y_true["labels"]),
                label_length=y_true["labels_length"],
                blank=self.blank,
                name=self.name,
            )
        else:
            loss = ctc_loss(
                y_pred=y_pred["logits"],
                input_length=y_pred["logits_length"],
                y_true=tf.sparse.from_dense(y_true["labels"]),
                label_length=y_true["labels_length"],
                blank=self.blank,
                name=self.name,
            )
        return tf.reduce_mean(loss)


class MaskedCE(tf.keras.losses.Loss):
    def __init__(self, name="masked_ce"):
        super().__init__(name=name)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

    def call(self, y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss_ = self.loss_fn(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_, axis=-1) / tf.reduce_sum(mask, axis=-1)


class CustomCrossEntropy(tf.keras.losses.Loss):
    def __init__(
        self,
        label_smoothing=0.0,
        ignore_class=0,
        from_dense=True,
        from_logits=True,
        **kwargs
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_class = ignore_class
        self.from_dense = from_dense
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        loss = custom_categorical_crossentropy(
            y_true=y_true,
            y_pred=y_pred,
            label_smoothing=self.label_smoothing,
            ignore_class=self.ignore_class,
            from_dense=self.from_dense,
            from_logits=self.from_logits,
        )
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=loss.dtype)
        return tf.reduce_sum(loss) / batch_size


class FocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(
        self,
        label_smoothing=0.0,
        gamma=2.0,
        alpha=0.25,
        ignore_class=0,
        from_dense=True,
        from_logits=True,
        **kwargs
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_class = ignore_class
        self.from_dense = from_dense
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        loss = focal_categorical_crossentropy(
            y_true=y_true,
            y_pred=y_pred,
            gamma=self.gamma,
            alpha=self.alpha,
            label_smoothing=self.label_smoothing,
            ignore_class=self.ignore_class,
            from_dense=self.from_dense,
            from_logits=self.from_logits,
        )
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=loss.dtype)
        return tf.reduce_sum(loss) / batch_size
