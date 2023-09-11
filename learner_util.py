## Learner.

import tensorflow as tf


## Learner Base Class
class CustomBaseModel(tf.keras.Model):
    def compile(self, criterion, optimizer, metrics):
        super().compile()
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric_list = metrics

    @property
    def metrics(self):
        return self.metric_list

    def train_step(self, data):
        x, labels = data
        batched_x_len = x.shape[1]  ## batched T for preventing NaN loss
        ## cause NxTxLxC or NxTxC, differ to x_len (individual length), x_len is for metric
        with tf.GradientTape() as tape:
            y_pred = {}
            y_true = {}
            predictions = self(x, training=True)

            y_pred["logits"] = predictions
            y_pred["logits_length"] = tf.fill(dims=(x.shape[0],), value=batched_x_len)
            y_true["labels"] = labels
            y_true["labels_length"] = tf.reduce_sum(
                tf.cast(labels != 0, tf.int32), axis=1
            )

            loss = self.criterion(y_true, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(
            labels,
            predictions,
            tf.reduce_sum(tf.cast(predictions._keras_mask, tf.int32), axis=-1),
        )

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, labels = data
        y_pred = {}
        y_true = {}
        predictions = self(x, training=False)

        y_pred["logits"] = predictions
        y_pred["logits_length"] = tf.reduce_sum(
            tf.cast(predictions._keras_mask, tf.int32), axis=-1
        )
        y_true["labels"] = labels
        y_true["labels_length"] = tf.reduce_sum(tf.cast(labels != 0, tf.int32), axis=1)

        loss = self.criterion(y_true, y_pred)
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(
            labels,
            predictions,
            y_pred["logits_length"],
        )
        return {m.name: m.result() for m in self.metrics}

class AWPInterCTC(CustomBaseModel):
    def __init__(self, *args, 
                 delta=0.1, eps=1e-6, 
                 start_step=0, clip_grads=1.0, 
                 inter_ctc_weight=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.eps = eps
        self.start_step = start_step
        self.clip_grads = clip_grads
        self.inter_ctc_weight = inter_ctc_weight
    
    def calc_loss(self, pred, y_true, batched_x_len):
        y_pred = {}
        y_pred["logits"] = pred
        y_pred["logits_length"] = tf.fill(dims=(pred.shape[0],), value=batched_x_len)
        loss = self.criterion(y_true, y_pred)
        return loss
    
    def inter_loss(self, pred, y_true, batched_x_len):
        loss = self.calc_loss(pred[0], y_true, batched_x_len)
        loss_inter = 0
        for i in range(1, len(pred)):
            loss_inter += self.calc_loss(pred[i], y_true, batched_x_len)
        loss_inter /= (len(pred) - 1)

        loss = (
            1 - self.inter_ctc_weight
        ) * loss + self.inter_ctc_weight * loss_inter
        return loss 

    def train_step_awp(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, labels = data
        batched_x_len = x.shape[1] 
        y_true = {}
        y_true["labels"] = labels
        y_true["labels_length"] = tf.reduce_sum(
            tf.cast(labels != 0, tf.int32), axis=1
        )

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.inter_loss(predictions, y_true, batched_x_len)

        params = self.trainable_variables
        params_gradients = tape.gradient(loss, self.trainable_variables)
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(
                self.delta * grad, tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps
            )
            self.trainable_variables[i].assign_add(delta)

        with tf.GradientTape() as tape2:
            predictions = self(x, training=True)
            new_loss = self.inter_loss(predictions, y_true, batched_x_len)

            if hasattr(self.optimizer, "get_scaled_loss"):
                new_loss = self.optimizer.get_scaled_loss(new_loss)

        gradients = tape2.gradient(new_loss, self.trainable_variables)
        if hasattr(self.optimizer, "get_unscaled_gradients"):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(
                self.delta * grad, tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps
            )
            self.trainable_variables[i].assign_sub(delta)
        if self.clip_grads > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_grads)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)  # train loss
        self.metrics[1].update_state(
            labels,
            predictions[0],
            tf.reduce_sum(tf.cast(predictions[0]._keras_mask, tf.int32), axis=-1),
        )
        return {m.name: m.result() for m in self.metrics}

    def train_step_base(self, data):
        x, labels = data
        batched_x_len = x.shape[1] 
        y_true = {}
        y_true["labels"] = labels
        y_true["labels_length"] = tf.reduce_sum(
            tf.cast(labels != 0, tf.int32), axis=1
        )

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.inter_loss(predictions, y_true, batched_x_len)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.clip_grads > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_grads)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)  # train loss
        self.metrics[1].update_state(
            labels,
            predictions[0],
            tf.reduce_sum(tf.cast(predictions[0]._keras_mask, tf.int32), axis=-1),
        )
        return {m.name: m.result() for m in self.metrics}
    
    def train_step(self, data):
        return tf.cond(
            self._train_counter < self.start_step,
            lambda: self.train_step_base(data),
            lambda: self.train_step_awp(data),
        )

    def test_step(self, data):
        x, labels = data
        y_pred = {}
        y_true = {}
        predictions = self(x, training=False)[0]

        y_pred["logits"] = predictions
        y_pred["logits_length"] = tf.reduce_sum(
            tf.cast(predictions._keras_mask, tf.int32), axis=-1
        )
        y_true["labels"] = labels
        y_true["labels_length"] = tf.reduce_sum(tf.cast(labels != 0, tf.int32), axis=1)

        loss = self.criterion(y_true, y_pred)
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(labels, predictions, y_pred["logits_length"])
        return {m.name: m.result() for m in self.metrics}


## AWP Base Learner
class AWP(CustomBaseModel):
    def __init__(self, *args, delta=0.1, eps=1e-4, start_step=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.eps = eps
        self.start_step = start_step

    def train_step_awp(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, labels = (data[0], data[1])
        input_len = x.shape[1]  ## cause NxTxLxC or NxTxC

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            y_pred = {}
            y_true = {}
            y_pred["logits"] = predictions
            if input_len == 384:
                y_pred["logits_length"] = tf.fill(dims=(x.shape[0],), value=384)
            else:
                y_pred["logits_length"] = tf.fill(dims=(x.shape[0],), value=80)
            y_true["labels"] = tf.sparse.from_dense(labels)
            loss = self.criterion(y_true, y_pred)
        params = self.trainable_variables
        params_gradients = tape.gradient(loss, self.trainable_variables)
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(
                self.delta * grad, tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps
            )
            self.trainable_variables[i].assign_add(delta)
        with tf.GradientTape() as tape2:
            predictions = self(x, training=True)
            y_pred = {}
            y_true = {}
            y_pred["logits"] = predictions
            if input_len == 384:
                y_pred["logits_length"] = tf.fill(dims=(x.shape[0],), value=384)
            else:
                y_pred["logits_length"] = tf.fill(dims=(x.shape[0],), value=80)
            y_true["labels"] = tf.sparse.from_dense(labels)
            new_loss = self.criterion(y_true, y_pred)
            if hasattr(self.optimizer, "get_scaled_loss"):
                new_loss = self.optimizer.get_scaled_loss(new_loss)

        gradients = tape2.gradient(new_loss, self.trainable_variables)
        if hasattr(self.optimizer, "get_unscaled_gradients"):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(
                self.delta * grad, tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps
            )
            self.trainable_variables[i].assign_sub(delta)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)  # train loss
        return {"loss": self.metrics[0].result()}

    def train_step(self, data):
        return tf.cond(
            self._train_counter < self.start_step,
            lambda: super(AWP, self).train_step(data),
            lambda: self.train_step_awp(data),
        )


## InterCTC Base Learner
class InterCTC(CustomBaseModel):
    def __init__(
        self, *args, inter_ctc_weight, inter_start_step, clip_grads, use_beam, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.inter_ctc_weight = inter_ctc_weight
        self.inter_start_step = inter_start_step
        self.clip_grads = clip_grads
        self.use_beam = use_beam

    def train_step_base(self, data):
        x, labels = data
        batched_x_len = x.shape[1]
        with tf.GradientTape() as tape:
            y_pred = {}
            y_true = {}
            predictions = self(x, training=True)[0]

            y_pred["logits"] = predictions
            y_pred["logits_length"] = tf.fill(dims=(x.shape[0],), value=batched_x_len)
            y_true["labels"] = labels
            y_true["labels_length"] = tf.reduce_sum(
                tf.cast(labels != 0, tf.int32), axis=1
            )

            loss = self.criterion(y_true, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(
            labels,
            predictions,
            tf.reduce_sum(tf.cast(predictions._keras_mask, tf.int32), axis=-1),
        )
        return {m.name: m.result() for m in self.metrics}

    def train_step_inter(self, data):
        x, labels = data
        batched_x_len = x.shape[1]
        with tf.GradientTape() as tape:
            y_pred_1 = {}
            y_pred_2 = {}
            y_pred_3 = {}
            y_true = {}

            preds = self(x, training=True)
            pred_1, pred_2, pred_3 = (
                preds[0],
                preds[1],
                preds[2],
            )
            y_pred_1["logits"] = pred_1
            y_pred_1["logits_length"] = tf.fill(dims=(x.shape[0],), value=batched_x_len)
            y_pred_2["logits"] = pred_2
            y_pred_2["logits_length"] = tf.fill(dims=(x.shape[0],), value=batched_x_len)
            y_pred_3["logits"] = pred_3
            y_pred_3["logits_length"] = tf.fill(dims=(x.shape[0],), value=batched_x_len)
            y_true["labels"] = labels
            y_true["labels_length"] = tf.reduce_sum(
                tf.cast(labels != 0, tf.int32), axis=1
            )
            loss_1 = self.criterion(y_true, y_pred_1)
            loss_2 = self.criterion(y_true, y_pred_2)
            loss_3 = self.criterion(y_true, y_pred_3)
            inter_ctc_loss = (loss_2 + loss_3) / 2
            loss = (
                1 - self.inter_ctc_weight
            ) * loss_1 + self.inter_ctc_weight * inter_ctc_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.clip_grads > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_grads)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss_1)
        self.metrics[1].update_state(
            labels,
            pred_1,
            tf.reduce_sum(tf.cast(pred_1._keras_mask, tf.int32), axis=-1),
        )
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return tf.cond(
            self._train_counter < self.inter_start_step,
            lambda: self.train_step_base(data),
            lambda: self.train_step_inter(data),
        )

    def test_step(self, data):
        x, labels = data
        y_pred = {}
        y_true = {}
        predictions = self(x, training=False)[0]

        y_pred["logits"] = predictions
        y_pred["logits_length"] = tf.reduce_sum(
            tf.cast(predictions._keras_mask, tf.int32), axis=-1
        )
        y_true["labels"] = labels
        y_true["labels_length"] = tf.reduce_sum(tf.cast(labels != 0, tf.int32), axis=1)

        loss = self.criterion(y_true, y_pred)
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(labels, predictions, y_pred["logits_length"])
        if self.use_beam:
            self.metrics[2].update_state(labels, predictions, y_pred["logits_length"])
        return {m.name: m.result() for m in self.metrics}
