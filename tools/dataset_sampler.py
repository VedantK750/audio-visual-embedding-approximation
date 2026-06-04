import pandas as pd
import random

# Load the CSV (no header)
df = pd.read_csv('vggsound.csv', header=None,
                 names=['youtube_id', 'start_seconds', 'label', 'split'])

# Manually choose semantically diverse classes
selected_classes = [
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
    "playing acoustic guitar"         # music
]


# Filter to selected classes
df_filtered = df[df['label'].isin(selected_classes)]

# --- Balance: take up to K clips per class per split ---
K = 200  # clips per class (so 15 classes x 200 = 3,000 train clips)

print(len(df_filtered))
print(df_filtered.head())

train_parts = []

for label, group in df_filtered[df_filtered['split'] == 'train'].groupby('label'):
    train_parts.append(
        group.sample(min(len(group), K), random_state=42)
    )

df_train = pd.concat(train_parts, ignore_index=True)

print(df_train.columns)
print(type(df_train))

test_parts = []
for label, group in df_filtered[df_filtered['split'] == 'test'].groupby('label'):
    test_parts.append(
        group.sample(min(len(group), 50), random_state=42)
    )
df_test = pd.concat(test_parts, ignore_index=True)


print(f"Train clips: {len(df_train)}")
print(f"Test clips:  {len(df_test)}")
print(df_train['label'].value_counts())

# Save filtered lists
df_train.to_csv('subset_train.csv', index=False)
df_test.to_csv('subset_test.csv', index=False)