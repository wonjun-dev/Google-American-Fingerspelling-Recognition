import os
import pandas as pd
from tqdm import tqdm
import json
import numpy as np

RAW_DATA = "/sources/dataset/raw_data"
PID_DATA = "/sources/dataset/pid_data_v6"

with open("/sources/dataset/raw_data/cols_to_idx.json", "r") as f:
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
# USE_COLS = [cols_to_idx[col] for col in USE_COLS]
NUM_LANDMARK = int(len(USE_COLS) // 2)


def main(dataset="train"):
    if dataset == "supplemental_metadata":
        lm_name = "supplemental_landmarks"
    elif dataset == "train":
        lm_name = "train_landmarks"

    df = pd.read_csv(os.path.join(RAW_DATA, f"{dataset}.csv"))
    pids = df["participant_id"].unique()
    pids.sort()

    print(f"# num pids: {len(pids)}")
    print(f"pids: {pids}")

    new_df = pd.DataFrame()
    total_except_cnt = 0
    total_not_found = 0
    for cnt, pid in enumerate(pids):
        print(f"{cnt+1}/{len(pids)}")
        pid_df = df[df["participant_id"] == pid]
        print(f" # of sequences in cur pid: {len(pid_df)}")

        new_parq = pd.DataFrame()
        prev_path = None
        except_cnt = 0
        not_found = 0
        for row in tqdm(pid_df.itertuples()):
            path = getattr(row, "path")
            seq_id = getattr(row, "sequence_id")
            phrase = getattr(row, "phrase")

            ### Make landmarks parquet ###
            if prev_path != path:
                parq = pd.read_parquet(os.path.join(RAW_DATA, path), columns=USE_COLS)

            if seq_id not in parq.index:
                except_cnt += 1
                not_found += 1
                prev_path = path
                continue

            lms = parq.loc[seq_id]

            if isinstance(lms, pd.Series):
                prev_path = path
                except_cnt += 1
                continue
            elif len(lms) < 2 * len(phrase) + 1:
                prev_path = path
                except_cnt += 1
                continue
            else:
                np_lms = np.array(lms)
                nan_cnt = np.all(np.isnan(np_lms), axis=1).sum()
                if nan_cnt == 0:
                    new_parq = pd.concat([new_parq, lms], ignore_index=False)
                    prev_path = path

                    ### Make info csv ###
                    row_info = pd.DataFrame(
                        {
                            "path": f"{lm_name}/{pid}.parquet",
                            "sequence_id": [seq_id],
                            "participant_id": [pid],
                            "phrase": [phrase],
                        }
                    )
                    new_df = pd.concat([new_df, row_info], ignore_index=True)
                elif nan_cnt > 0:
                    if len(lms) - nan_cnt < 2 * len(phrase) + 1:
                        prev_path = path
                        except_cnt += 1
                        continue
                    else:
                        new_parq = pd.concat([new_parq, lms], ignore_index=False)
                        prev_path = path

                        ### Make info csv ###
                        row_info = pd.DataFrame(
                            {
                                "path": f"{lm_name}/{pid}.parquet",
                                "sequence_id": [seq_id],
                                "participant_id": [pid],
                                "phrase": [phrase],
                            }
                        )
                        new_df = pd.concat([new_df, row_info], ignore_index=True)

        # Sanity check
        print(f"# of saved sequences: {len(new_parq.index.unique())}")
        assert len(new_parq.index.unique()) == len(pid_df) - except_cnt
        # Save parquet
        new_parq.to_parquet(os.path.join(PID_DATA, lm_name, f"{pid}.parquet"))
        total_except_cnt += except_cnt
        total_not_found += not_found

    # Save info csv
    new_df.to_csv(os.path.join(PID_DATA, f"{dataset}.csv"), index=False)
    print(f"# of new df sequences: {len(new_df)}")
    print(f"# of origin df sequences: {len(df)}")
    print(f"# of except sequences: {total_except_cnt}")
    print(f"# of not found sequences: {total_not_found}")
    assert len(new_df) == len(df) - total_except_cnt


if __name__ == "__main__":
    main(dataset="train")
    main(dataset="supplemental_metadata")
