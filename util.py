## LR, Callback, etc.
import os
import random
import tensorflow as tf
import numpy as np

from datetime import datetime


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)


def set_log_ckpt_path(args, user="wj"):
    log_dir = os.path.join(args.log_path, user, args.exp_name)
    ckpt_dir = os.path.join(args.ckpt_path, user, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    val_fold = args.val_fold
    if val_fold != -1:
        log_path = os.path.join(log_dir, str(val_fold), time_stamp)
        ckpt_path = os.path.join(ckpt_dir, str(val_fold), time_stamp)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
    elif val_fold == -1:
        log_path = os.path.join(log_dir, "all", time_stamp)
        ckpt_path = os.path.join(ckpt_dir, "all", time_stamp)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
    print("log_path: ", log_path)
    print("ckpt_path: ", ckpt_path)

    return log_path, ckpt_path


def lr_warmup_cosine_decay(
    global_step, warmup_steps, hold=0, total_steps=0, start_lr=0.0, target_lr=3e-4
):
    learning_rate = (
        0.5
        * target_lr
        * (
            1
            + tf.cos(
                tf.constant(np.pi)
                * (global_step - warmup_steps - hold)
                / float(total_steps - warmup_steps - hold)
            )
        )
    )
    warmup_lr = target_lr * (global_step / warmup_steps)

    if hold > 0:
        learning_rate = tf.where(
            global_step > warmup_steps + hold, learning_rate, target_lr
        )

    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = lr_warmup_cosine_decay(
            global_step=step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            hold=self.hold,
        )

        return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")


class EditScore(tf.keras.metrics.Mean):
    def __init__(self, name="edit_score", use_beam=False, **kwargs):
        super(EditScore, self).__init__(name=name, **kwargs)
        self.use_beam = use_beam

    def get_edit_score(self, y_true, y_pred, y_pred_length):
        if not self.use_beam:
            decoded, _ = tf.nn.ctc_greedy_decoder(
                tf.transpose(y_pred, perm=[1, 0, 2]),
                sequence_length=y_pred_length,
                merge_repeated=True,
                blank_index=0,
            )
        else:
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                tf.transpose(y_pred, perm=[1, 0, 2]),
                sequence_length=y_pred_length,
                beam_width=5,
                top_paths=1,
            )

        dist = 1 - (
            tf.edit_distance(
                tf.cast(decoded[0], dtype=tf.int32),
                tf.sparse.from_dense(y_true),
                normalize=True,
            )
        )
        return tf.reduce_mean(dist)

    def update_state(self, y_true, y_pred, y_pred_length, sample_weight=None):
        score = self.get_edit_score(y_true, y_pred, y_pred_length)
        return super().update_state(score, sample_weight)


class EditScoreDec(tf.keras.metrics.Mean):
    def __init__(self, name="edit_score", **kwargs):
        super(EditScoreDec, self).__init__(name=name, **kwargs)

    def get_edit_score(self, y_true, y_pred):
        decoded = tf.argmax(y_pred, axis=-1)
        first_eos = tf.argmax(tf.equal(decoded, 61), axis=1)
        indices = tf.range(tf.shape(decoded)[1], dtype=tf.int64)
        mask = tf.less(indices, tf.expand_dims(first_eos, axis=-1))
        decoded = tf.where(mask, decoded, 0)
        y_true = tf.where(
            tf.math.equal(y_true, 61), tf.constant(0, dtype=tf.int32), y_true
        )
        dist = 1 - (
            tf.edit_distance(
                tf.sparse.from_dense(tf.cast(decoded, dtype=tf.int32)),
                tf.sparse.from_dense(y_true),
                normalize=True,
            )
        )
        return tf.reduce_mean(dist)

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = self.get_edit_score(y_true, y_pred)
        return super().update_state(score, sample_weight)


from keras.utils import io_utils
from keras.utils import tf_utils


class TerminateOnNaNV2(tf.keras.callbacks.TerminateOnNaN):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("enc_loss")
        if loss is not None:
            loss = tf_utils.sync_to_numpy_or_python_type(loss)
            if np.isnan(loss) or np.isinf(loss):
                io_utils.print_msg(f"Batch {batch}: Invalid loss, terminating training")
                self.model.stop_training = True


class GradientLogger(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        with tf.GradientTape() as tape:
            y_pred = self.model(
                self.model.train_on_batch._inputs, training=True
            )  # Compute forward pass
            loss = self.model.compiled_loss._losses  # Retrieve the loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        for i, grad in enumerate(gradients):
            tf.summary.histogram(
                "grad_{}".format(i), grad, step=self.model.optimizer.iterations
            )


class WarmCosineRestart(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, target_lr, warmup_steps, total_steps, decay_step, t_mul, m_mul):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=target_lr,
            first_decay_steps=decay_step,
            t_mul=t_mul,
            m_mul=m_mul,
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.target_lr * (step / self.warmup_steps),
            lambda: self.scheduler(step - self.warmup_steps),
        )


class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Unified single-cycle learning rate scheduler for tensorflow.
    2022 Hoyeol Sohn <hoeyol0730@gmail.com>
    """

    def __init__(
        self,
        lr=1e-4,
        epochs=10,
        steps_per_epoch=100,
        steps_per_update=1,
        resume_epoch=0,
        decay_epochs=10,
        sustain_epochs=0,
        warmup_epochs=0,
        lr_start=0,
        lr_min=0,
        warmup_type="linear",
        decay_type="cosine",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr = float(lr)
        self.epochs = float(epochs)
        self.steps_per_update = float(steps_per_update)
        self.resume_epoch = float(resume_epoch)
        self.steps_per_epoch = float(steps_per_epoch)
        self.decay_epochs = float(decay_epochs)
        self.sustain_epochs = float(sustain_epochs)
        self.warmup_epochs = float(warmup_epochs)
        self.lr_start = float(lr_start)
        self.lr_min = float(lr_min)
        self.decay_type = decay_type
        self.warmup_type = warmup_type

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        total_steps = self.epochs * self.steps_per_epoch
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        sustain_steps = self.sustain_epochs * self.steps_per_epoch
        decay_steps = self.decay_epochs * self.steps_per_epoch

        if self.resume_epoch > 0:
            step = step + self.resume_epoch * self.steps_per_epoch

        step = tf.cond(step > decay_steps, lambda: decay_steps, lambda: step)
        step = tf.math.truediv(step, self.steps_per_update) * self.steps_per_update

        warmup_cond = step < warmup_steps
        decay_cond = step >= (warmup_steps + sustain_steps)

        if self.warmup_type == "linear":
            lr = tf.cond(
                warmup_cond,
                lambda: tf.math.divide_no_nan(self.lr - self.lr_start, warmup_steps)
                * step
                + self.lr_start,
                lambda: self.lr,
            )
        elif self.warmup_type == "exponential":
            factor = tf.pow(self.lr_start, 1 / warmup_steps)
            lr = tf.cond(
                warmup_cond,
                lambda: (self.lr - self.lr_start) * factor ** (warmup_steps - step)
                + self.lr_start,
                lambda: self.lr,
            )
        elif self.warmup_type == "cosine":
            lr = tf.cond(
                warmup_cond,
                lambda: 0.5
                * (self.lr - self.lr_start)
                * (1 + tf.cos(3.14159265359 * (warmup_steps - step) / warmup_steps))
                + self.lr_start,
                lambda: self.lr,
            )
        else:
            raise NotImplementedError

        if self.decay_type == "linear":
            lr = tf.cond(
                decay_cond,
                lambda: self.lr
                + (self.lr_min - self.lr)
                / (decay_steps - warmup_steps - sustain_steps)
                * (step - warmup_steps - sustain_steps),
                lambda: lr,
            )
        elif self.decay_type == "exponential":
            factor = tf.pow(
                self.lr_min, 1 / (decay_steps - warmup_steps - sustain_steps)
            )
            lr = tf.cond(
                decay_cond,
                lambda: (self.lr - self.lr_min)
                * factor ** (step - warmup_steps - sustain_steps)
                + self.lr_min,
                lambda: lr,
            )
        elif self.decay_type == "cosine":
            lr = tf.cond(
                decay_cond,
                lambda: 0.5
                * (self.lr - self.lr_min)
                * (
                    1
                    + tf.cos(
                        3.14159265359
                        * (step - warmup_steps - sustain_steps)
                        / (decay_steps - warmup_steps - sustain_steps)
                    )
                )
                + self.lr_min,
                lambda: lr,
            )
        else:
            raise NotImplementedError

        return lr

    def plot(self):
        import matplotlib.pyplot as plt

        step = max(
            1, int(self.epochs * self.steps_per_epoch) // 1000
        )  # 1 for total_steps < 1000, total_steps//1000 else
        eps = list(range(0, int(self.epochs * self.steps_per_epoch), step))
        learning_rates = [self(x) for x in eps]
        plt.scatter(eps, learning_rates, 2)
        plt.show()


class EarlyStopAfterNEpochs(tf.keras.callbacks.Callback):
    def __init__(self, stop_after_epochs=60):
        super(EarlyStopAfterNEpochs, self).__init__()
        self.stop_after_epochs = stop_after_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 >= self.stop_after_epochs:
            print(f"\nReached {self.stop_after_epochs} epochs. Stopping training.")
            self.model.stop_training = True
