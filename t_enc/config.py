import argparse

parser = argparse.ArgumentParser(description="Google ASLFR")


# ========================= General Configs ==========================
parser.add_argument("--gpu", default=7, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--exp_name", default="baseline", type=str)
parser.add_argument(
    "--max_len",
    default=384,
    type=int,
    help="Used for data truncatino and model dimension",
)
parser.add_argument(
    "--transfer", action="store_true", help="initialize with pretrained weights"
)
parser.add_argument("--data_ver", default="v4", type=str)
parser.add_argument("--user", default="wj", type=str)
# ========================= Data Configs ==========================
parser.add_argument(
    "--data_type", default="train", type=str, help="train, supplemental"
)
parser.add_argument("--val_fold", default=1, type=int, help="-1 for full train")
# ========================= Augmentation Configs ==========================
parser.add_argument("--augment", action="store_true")
parser.add_argument("--use_flip_lr", action="store_true")
parser.add_argument("--use_affine", action="store_true")
parser.add_argument("--use_resample", action="store_true")
parser.add_argument("--use_reverse", action="store_true")
parser.add_argument("--rate", default=0.2, type=float)
# ========================= Preproc Configs ==========================
parser.add_argument("--use_major", action="store_true")  # 키면 우세손만 사용
# ========================= Model Configs ==========================
parser.add_argument("--num_heads", default=4, type=int)
parser.add_argument(
    "--kernel_size",
    default=11,
    type=int,
)
parser.add_argument(
    "--hidden_dim",
    default=192,
    type=int,
)
parser.add_argument("--class_num", default=60, type=int, help="Number of class")
parser.add_argument(
    "--pretrain_path",
    default="/sources/dataset/ckpts/wj/enc_baseline_lip_supp/all/2023-08-13T15-36-13/model_last_ep.h5",
    help="pretrain model ckpt path",
)
parser.add_argument(
    "--freeze_feat", action="store_true", help="freeze feature extractor layers"
)
parser.add_argument("--cbam_ks", default=11, type=int, help="cbam module kernel size")
parser.add_argument("--expand_ratio", default=4, type=int)

# ========================= Learning Configs ==========================
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--lr_scheduler", action="store_true")
parser.add_argument("--start_lr", default=0.0, type=float)
parser.add_argument("--target_lr", default=3e-3, type=float)
parser.add_argument("--warmup_epoch", default=10, type=int)
parser.add_argument("--hold", default=0, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--sync_period", default=5, type=int)
parser.add_argument("--dropout", default=0.2, type=float)
parser.add_argument("--late_dropout", default=0.5, type=float)
parser.add_argument("--late_dropout_ver", default="v1", type=str)
parser.add_argument("--late_dropout_step", default=15, type=int)
parser.add_argument("--inter_ctc_weight", default=0.5, type=float)
parser.add_argument("--inter_ctc_step", default=0, type=int)
parser.add_argument("--clip_grads", default=0.0, type=float)
parser.add_argument("--focal", action="store_true")
parser.add_argument("--awp_lambda", default=0.2, type=float, help="AWP delta")
parser.add_argument("--use_awp", action="store_true")
parser.add_argument("--awp_step", default=15, type=int, help="awp start epoch")

parser.add_argument("--use_beam", action="store_true")
parser.add_argument("--use_stemconv", action="store_true")
parser.add_argument("--supptrain", action="store_true")
parser.add_argument("--fp16", action="store_true")
# ========================= Runtime Configs ==========================
parser.add_argument("--log_path", default="/sources/dataset/logs", type=str)
parser.add_argument("--ckpt_path", default="/sources/dataset/ckpts", type=str)
