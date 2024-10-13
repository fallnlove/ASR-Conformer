from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, dataset_dir, *args, **kwargs):
        data = []
        audio_path = Path(dataset_dir) / "audio"
        transcription_dir = Path(dataset_dir) / "transcriptions"

        for path in audio_path.iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)

                t_info = torchaudio.info(str(path))
                length = t_info.num_frames / t_info.sample_rate
                entry["audio_len"] = length

                if transcription_dir.exists():
                    transc_path = transcription_dir / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip().lower()
                else:
                    entry["text"] = ""
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
