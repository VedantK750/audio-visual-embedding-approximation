import os
from PIL import Image
from more_itertools import sample
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio 


class VGGSoundDataset(Dataset):
    def __init__(self, root_dir, split="train"):

        self.root_dir = root_dir
        self.split = split

        self.teacher_root = os.path.join(
            root_dir,
            split,
            "teacher_embeddings"
        )
        print(self.teacher_root)

        self.samples = []
        for label in os.listdir(self.teacher_root):

            label_dir = os.path.join(
                self.teacher_root,
                label
            )

            if not os.path.isdir(label_dir):
                continue

            for file in os.listdir(label_dir):

                if not file.endswith(".npy"):
                    continue

                clip_id = file.replace(".npy", "")

                teacher_path = os.path.join(
                    label_dir,
                    file
                )

                frame_dir = os.path.join(
                    root_dir,
                    split,
                    "frames",
                    label,
                    clip_id
                )

                audio_path = os.path.join(
                    root_dir,
                    split,
                    "audio",
                    label,
                    f"{clip_id}.wav"
                )

                teacher_vision_path = os.path.join(
                root_dir,
                split,
                "teacher_embeddings_vision",
                label,
                file
                )

                teacher_audio_path = os.path.join(
                    root_dir,
                    split,
                    "teacher_embeddings_audio",
                    label,
                    file
                )

                if not (
                    os.path.exists(frame_dir)
                    and os.path.exists(audio_path)
                    and os.path.exists(teacher_path)
                    and os.path.exists(teacher_vision_path)
                    and os.path.exists(teacher_audio_path)
                ):
                    continue

                self.samples.append(
                {
                    "label": label,
                    "clip_id": clip_id,
                    "teacher_path": teacher_path,
                    "teacher_vision_path": teacher_vision_path,
                    "teacher_audio_path": teacher_audio_path,
                    "frame_dir": frame_dir,
                    "audio_path": audio_path
                }
            )

        print(
            f"Loaded {len(self.samples)} samples "
            f"from {split}"
        )
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        # load frames
        sample = self.samples[idx]

        frame_dir = sample["frame_dir"]
        audio_path = sample["audio_path"]
        teacher_path = sample["teacher_path"]
        teacher_vision_path = sample["teacher_vision_path"]
        teacher_audio_path = sample["teacher_audio_path"]
        frame_paths = sorted(
        [
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.endswith(".jpg")
        ]
    )
        
        images = []

        for frame_path in frame_paths:
            # print(Image)
            images.append(
                Image.open(frame_path)
                .convert("RGB")
            )
        assert len(images) == 5, (
        f"Expected 5 frames, got {len(images)} "
        f"for {frame_dir}"
        )
        # load audio
        waveform, sr = torchaudio.load(
            audio_path
        )

        teacher = torch.tensor(
            np.load(teacher_path),
            dtype=torch.float32
        )

        teacher_vision = torch.tensor(
            np.load(teacher_vision_path),
            dtype=torch.float32
        )

        teacher_audio = torch.tensor(
            np.load(teacher_audio_path),
            dtype=torch.float32
        )

        return {
        "images": images,
        "waveform": waveform,
        "sample_rate": sr,
        "teacher": teacher,
        "teacher_vision": teacher_vision,
        "teacher_audio": teacher_audio,
        "label": sample["label"],
        "clip_id": sample["clip_id"]
        }


def main():
    dataset = VGGSoundDataset(root_dir="processed_vggsound", split="test")
    sample = dataset[0]
    print(sample["images"][0].size)
    print(sample["waveform"].shape)
    print(sample["teacher"].shape)  

if __name__ == "__main__":
    main()