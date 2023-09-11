import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("/sources/dataset/refactor_v3_hdist")

import tensorflow as tf
import tensorflow_addons as tfa

from config import parser
from dataloader import get_tfrec_dataset
from model import get_model

from util import (
    seed_everything,
    set_log_ckpt_path,
    WarmUpCosineDecay,
    EditScore,
    EarlyStopAfterNEpochs,
)
from loss_util import CTCLoss
from data_util import (
    prepare_data,
    TRAIN_TRAIN_LEN,
    TRAIN_VAL_LEN,
    SUPP_TRAIN_LEN,
    SUPP_VAL_LEN,
)
from learner_util import AWPInterCTC


args = parser.parse_args()
gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[args.gpu], "GPU")

seed_everything(args.seed)
log_path, ckpt_path = set_log_ckpt_path(args, user=args.user)  # "wj" or "dj"
# ========================= Callbacks ==========================
if args.val_fold == -1:
    monitor = ["loss", "score"]
else:
    monitor = ["val_loss", "val_score"]

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_path, update_freq="epoch"
)
logger_callback = tf.keras.callbacks.CSVLogger(
    log_path + "/log.csv", separator=",", append=False
)
ckpt_callback_1 = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path + "/best_loss.h5",
    monitor=monitor[0],
    save_best_only=True,
    save_weights_only=True,
    mode="min",
)
ckpt_callback_2 = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path + "/best_score.h5",
    monitor=monitor[1],
    save_best_only=True,
    save_weights_only=True,
    mode="max",
)
# earlystop_callback = EarlyStopAfterNEpochs(stop_after_epochs=60)
nan_callback = tf.keras.callbacks.TerminateOnNaN()
callbacks = [
    tensorboard_callback,
    logger_callback,
    ckpt_callback_1,
    ckpt_callback_2,
    # earlystop_callback,
    nan_callback,
]
# ========================= Data Configs ==========================
val_fold = args.val_fold
max_phrase_len = 31 if args.data_type == "train" else 43
train_tfrecs, val_tfrecs = prepare_data(args.data_type, val_fold, args.data_ver)
train_dataset = get_tfrec_dataset(
    train_tfrecs,
    batch_size=args.batch_size,
    max_len=args.max_len,
    max_phrase_len=max_phrase_len,
    augment=args.augment,
    use_flip_lr=args.use_flip_lr,
    use_affine=args.use_affine,
    use_resample=args.use_resample,
    use_reverse=args.use_reverse,
    rate=args.rate,
    use_major=args.use_major,
    repeat=True,
    shuffle=70000,
    drop_remainder=True,
)
if val_tfrecs is not None:
    val_dataset = get_tfrec_dataset(
        val_tfrecs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_phrase_len=max_phrase_len,
        augment=False,
        use_major=args.use_major,
        repeat=False,
        shuffle=0,
        drop_remainder=False,
    )
else:
    val_dataset = None

if val_tfrecs is not None:
    step_per_epoch = (
        TRAIN_TRAIN_LEN // args.batch_size
        if args.data_type == "train"
        else SUPP_TRAIN_LEN // args.batch_size
    )
else:
    step_per_epoch = (
        (TRAIN_TRAIN_LEN + TRAIN_VAL_LEN) // args.batch_size
        if args.data_type == "train"
        else (SUPP_TRAIN_LEN + SUPP_VAL_LEN) // args.batch_size
    )
total_step = step_per_epoch * args.epochs
# ========================= Optimizer ==========================
# if args.lr_scheduler_type == "cosinedecay":
scheduler = WarmUpCosineDecay(
    start_lr=args.start_lr,
    target_lr=args.target_lr,
    warmup_steps=step_per_epoch * args.warmup_epoch,
    total_steps=total_step,
    hold=args.hold,
)
print(f"Using CosineDecay with Warmup {args.warmup_epoch} epoch")
# elif args.lr_scheduler_type == "swa":
#     print('swa scheduler')
#     scheduler = WarmUpCosineSWA(
#         start_lr=args.start_lr,
#         target_lr=args.target_lr,
#         warmup_steps=step_per_epoch * args.warmup_epoch,
#         total_steps=total_step,
#         hold=args.hold,
#         swa_step=step_per_epoch * args.swa_epoch
#     )

optimizer = tfa.optimizers.RectifiedAdam(
    learning_rate=scheduler, weight_decay=args.weight_decay
)
optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=5)


# ========================= Model ==========================

model = get_model(
    max_len=args.max_len,
    hidden_dim=args.hidden_dim,
    kernel_size=args.kernel_size,
    cbam_kernel_size=args.cbam_ks,
    num_heads=args.num_heads,
    expand_ratio=args.expand_ratio,
    dropout=args.dropout,
    late_dropout=args.late_dropout,
    late_dropout_start=step_per_epoch * args.late_dropout_step,
    use_major=args.use_major,
    class_num=args.class_num,
)
print(model.summary())

if args.transfer:
    reinit_layers = [
        # "shared_ln",
        "shared_cls",
        "shared_cls_dense",
        "late_dropout_v2",
        "tf.nn.softmax",
        "tf.nn.softmax_1",
        "add_1",
        "add_2",
    ]

    # load pretrained model
    pretrain_weights = {}
    model.load_weights(args.pretrain_path)
    for layer in model.layers:
        if layer.name not in reinit_layers:
            pretrain_weights[layer.name] = layer.get_weights()

    # reinit layers, set pretrained weights
    model = get_model(
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        cbam_kernel_size=args.cbam_ks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        late_dropout=args.late_dropout,
        late_dropout_start=step_per_epoch * args.late_dropout_step,
        use_major=args.use_major,
        class_num=args.class_num,
    )
    for layer_name, layer_weights in pretrain_weights.items():
        model.get_layer(layer_name).set_weights(layer_weights)

    print(f"Pretrained layers: {pretrain_weights.keys()}")
    print(f"Reinit layers: {reinit_layers}")


model = AWPInterCTC(
    inputs=model.input,
    outputs=model.output,
    inter_ctc_weight=args.inter_ctc_weight,
    start_step=step_per_epoch * args.awp_step,
    clip_grads=args.clip_grads,
    delta=args.awp_lambda,
)

model.compile(
    criterion=CTCLoss(blank=0),
    optimizer=optimizer,
    metrics=[
        tf.keras.metrics.Mean("loss"),
        EditScore(name="score", use_beam=False),
    ],
)
# ========================= Train ==========================
model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=step_per_epoch,
    epochs=args.epochs,
    callbacks=callbacks,
)

if args.val_fold == -1:
    model.save_weights(ckpt_path + "model_last_ep.h5")
