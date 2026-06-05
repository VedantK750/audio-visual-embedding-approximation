"""Sample a balanced, semantically diverse 15-class subset of VGGSound.

VGGSound (Chen et al., 2020): ~200k ~10s YouTube clips, 309 sound classes where
the source is visible. Full dataset + CSV: https://www.robots.ox.ac.uk/~vgg/data/vggsound/

We hand-pick 15 classes spanning distinct coarse categories (animal, transport,
weather, nature, human, indoor, music) so retrieval/probe metrics are not
dominated by visually or acoustically near-duplicate classes, then balance per
class: up to 200 train and 50 test clips each (target 3000 train / 750 test;
the usable set is smaller after download/extraction failures).

Input:  vggsound.csv  (no header: youtube_id, start_seconds, label, split)
Output: subset_train.csv, subset_test.csv
"""

import pandas as pd

TRAIN_PER_CLASS = 200
TEST_PER_CLASS = 50
SEED = 42

# Semantically diverse classes, one or two per coarse category.
SELECTED_CLASSES = [
    "dog barking",                    # animal
    "cat meowing",                    # animal
    "bird chirping, tweeting",        # animal
    "helicopter",                     # aircraft
    "train horning",                  # transportation
    "car engine knocking",            # transportation
    "thunder",                        # weather
    "raining",                        # weather
    "ocean burbling",                 # nature
    "waterfall burbling",             # nature
    "people crowd",                   # human activity
    "female speech, woman speaking",  # speech
    "typing on computer keyboard",    # office / indoor activity
    "playing piano",                  # music
    "playing acoustic guitar",        # music
]


def balanced_sample(df, split, k):
    parts = []
    for _, group in df[df["split"] == split].groupby("label"):
        parts.append(group.sample(min(len(group), k), random_state=SEED))
    return pd.concat(parts, ignore_index=True)


def main():
    df = pd.read_csv(
        "vggsound.csv", header=None,
        names=["youtube_id", "start_seconds", "label", "split"],
    )
    df = df[df["label"].isin(SELECTED_CLASSES)]

    df_train = balanced_sample(df, "train", TRAIN_PER_CLASS)
    df_test = balanced_sample(df, "test", TEST_PER_CLASS)

    print(f"Train clips: {len(df_train)}")
    print(f"Test clips:  {len(df_test)}")
    print(df_train["label"].value_counts())

    df_train.to_csv("subset_train.csv", index=False)
    df_test.to_csv("subset_test.csv", index=False)


if __name__ == "__main__":
    main()
