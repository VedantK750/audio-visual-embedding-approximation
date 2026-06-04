import os
import random 
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback

import pandas as pd

import numpy as np
import torch

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


# ==========================
# CONFIG
# ==========================

MAX_WORKERS = 1
TIMEOUT = 300
RETRIES = 3
# TEST_SIZE = 100

# FRAME_TIMES = [1, 3, 5, 7, 9]

ROOT_DIR = "processed_vggsound_testing"

TEACHER_DIR = os.path.join(
    ROOT_DIR,
    "teacher_embeddings"
)

os.makedirs(
    TEACHER_DIR,
    exist_ok=True
)


VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

imagebind_model_instance = (
    imagebind_model.imagebind_huge(
        pretrained=True
    )
)

imagebind_model_instance.eval()
imagebind_model_instance.to(device)


@torch.no_grad()
def generate_imagebind_teacher(
    frame_dir,
    audio_path,
    teacher_path,
    vision_teacher_path,
    audio_teacher_path
):

    frame_paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(".jpg")
    ])

    vision_inputs = (
        data.load_and_transform_vision_data(
            frame_paths,
            device
        )
    )

    audio_inputs = (
        data.load_and_transform_audio_data(
            [audio_path],
            device
        )
    )

    embeddings = (
        imagebind_model_instance(
            {
                ModalityType.VISION:
                    vision_inputs,

                ModalityType.AUDIO:
                    audio_inputs
            }
        )
    )


    print(embeddings[ModalityType.VISION].shape)



    vision_emb = embeddings[
        ModalityType.VISION
    ].mean(dim=0)

    audio_emb = embeddings[
        ModalityType.AUDIO
    ].squeeze(0)

    teacher_emb = torch.cat(
        [
            vision_emb,
            audio_emb
        ],
        dim=0
    )

    np.save(
    vision_teacher_path,
    vision_emb.cpu().numpy()
    )

    np.save(
        audio_teacher_path,
        audio_emb.cpu().numpy()
    )

    np.save(
        teacher_path,
        teacher_emb.cpu().numpy()
    )

os.makedirs(VIDEOS_DIR, exist_ok=True)


# ==========================
# HELPERS
# ==========================

def sanitize_label(label):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", label)

def get_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ],
        capture_output=True,
        text=True,
        check=True
    )

    return float(result.stdout.strip())


def download_video(youtube_id, start_sec, output_path):
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    end_sec = start_sec + 10

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-f",
        "bv*+ba/b",
        "--download-sections",
        f"*{start_sec}-{end_sec}",
        "--force-keyframes-at-cuts",
        "--merge-output-format",
        "mp4",
        "-o",
        output_path,
        url
    ]
    time.sleep(
    random.uniform(0.5, 2.0)
    )   
    result = subprocess.run(
    cmd,
    timeout=TIMEOUT,
    capture_output=True,
    text=True
)

    if result.returncode != 0:
        raise Exception(result.stderr.strip())


def extract_frames(video_path, frame_dir):

    os.makedirs(frame_dir, exist_ok=True)

    duration = get_duration(video_path)

    # avoid very end of video
    max_time = max(1.0, duration - 0.5)

    frame_times = [
        max_time * 0.1,
        max_time * 0.3,
        max_time * 0.5,
        max_time * 0.7,
        max_time * 0.9
    ]

    for idx, t in enumerate(frame_times):

        frame_path = os.path.join(
            frame_dir,
            f"frame_{idx}.jpg"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(t),
            "-i",
            video_path,
            "-frames:v",
            "1",
            frame_path
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    

def extract_audio(video_path, audio_path):

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        audio_path
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


# ==========================
# PROCESS ONE CLIP
# ==========================

def process_clip(row):

    youtube_id = row["youtube_id"]
    start_sec = int(row["start_seconds"])
    label = row["label"]
    split = row["split"]

    safe_label = sanitize_label(label)

    video_path = os.path.join(
        VIDEOS_DIR,
        f"{youtube_id}_{start_sec}.mp4"
    )

    frame_dir = os.path.join(
    ROOT_DIR,
    split,
    "frames",
    safe_label,
    f"{youtube_id}_{start_sec}"
    )

    audio_path = os.path.join(
    ROOT_DIR,
    split,
    "audio",
    safe_label,
    f"{youtube_id}_{start_sec}.wav"
)

    os.makedirs(
        os.path.dirname(audio_path),
        exist_ok=True
    )

    teacher_path = os.path.join(
    ROOT_DIR,
    split,
    "teacher_embeddings",
    safe_label,
    f"{youtube_id}_{start_sec}.npy"
    )

    vision_teacher_path = os.path.join(
    ROOT_DIR,
    split,
    "teacher_embeddings_vision",
    safe_label,
    f"{youtube_id}_{start_sec}.npy"
)

    audio_teacher_path = os.path.join(
        ROOT_DIR,
        split,
        "teacher_embeddings_audio",
        safe_label,
        f"{youtube_id}_{start_sec}.npy"
    )

    for path in [
    teacher_path,
    vision_teacher_path,
    audio_teacher_path
    ]:
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )
    if (
    os.path.exists(teacher_path)
    and os.path.exists(vision_teacher_path)
    and os.path.exists(audio_teacher_path)
    ):
        return {
        "status": "already_done",
        "youtube_id": youtube_id,
        "label": label,
        "split": split
        }

    for attempt in range(RETRIES):

        try:

            download_video(
                youtube_id,
                start_sec,
                video_path
            )

            extract_frames(
                video_path,
                frame_dir
            )

            extract_audio(
                video_path,
                audio_path
            )

            generate_imagebind_teacher(
            frame_dir,
            audio_path,
            teacher_path,
            vision_teacher_path,
            audio_teacher_path
            )

            # delete mp4 after extraction
            if os.path.exists(video_path):
                os.remove(video_path)

            return {
                "status": "success",
                "youtube_id": youtube_id,
                "label": label,
                "split": split,
                "frame_dir": frame_dir,
                "audio_path": audio_path
            }

        except Exception as e:

            print("\n" + "=" * 80)
            print("FAILED CLIP")
            print("youtube_id:", youtube_id)
            print("label:", label)
            print("=" * 80)

            traceback.print_exc()

            if attempt == RETRIES - 1:

                return {
                    "status": "failed",
                    "youtube_id": youtube_id,
                    "label": label,
                    "split": split,
                    "reason": str(e)
                }

    return None


# ==========================
# MAIN
# ==========================

def main():

    train_df = pd.read_csv("subset_train.csv")
    test_df = pd.read_csv("subset_test.csv")

    df = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    # df = df.sample(
    #     n=min(TEST_SIZE, len(df)),
    #     random_state=42
    # )
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(f"Processing {len(df)} clips")

    successful = []
    failed = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_clip,
                row
            )
            for _, row in df.iterrows()
        ]

        completed = 0

        for future in as_completed(futures):

            result = future.result()

            completed += 1

            if result["status"] == "success":

                successful.append(result)

                print(
                    f"[{completed}/{len(df)}] "
                    f"SUCCESS "
                    f"{result['youtube_id']}"
                )
            elif result["status"] == "already_done":

                print(
                    f"[{completed}/{len(df)}] "
                    f"SKIPPED "
                    f"{result['youtube_id']}"
                )
                
            else:

                failed.append(result)

                print(
                    f"[{completed}/{len(df)}] "
                    f"FAILED "
                    f"{result['youtube_id']}"
                )

    pd.DataFrame(successful).to_csv(
        "successful.csv",
        index=False
    )

    pd.DataFrame(failed).to_csv(
        "failed.csv",
        index=False
    )

    print()
    print(f"Success: {len(successful)}")
    print(f"Failed : {len(failed)}")

    success_rate = (
    len(successful)
    /
    (len(successful) + len(failed))
)

    print(
        f"Success rate: "
        f"{success_rate*100:.2f}%"
    )

    success_df = pd.read_csv("successful.csv")
    success_stats = success_df["label"].value_counts(normalize=True).sort_values()
    print(success_stats)
    success_stats.to_csv("success_stats.csv")

if __name__ == "__main__":
    main()