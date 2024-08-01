import funcy
import os
import torch

from glob import glob
from torch.utils.data import Dataset

from chatgptaylor.tokenizer import Tokenizer


class LyricsDataset(Dataset):
    def __init__(self, datadir, context_length=8):
        self.lyrics_filepaths = glob(os.path.join(datadir, "*", "*.txt"))

        self.tokenizer = Tokenizer.load()

        self.context_length = context_length

    def __len__(self):
        return len(self.lyrics_filepaths)

    def __getitem__(self, item):
        filepath = self.lyrics_filepaths[item]
        song_title, _ = os.path.basename(filepath).rsplit(".", maxsplit=1)
        artist = os.path.basename(os.path.dirname(filepath))

        with open(filepath) as f:
            lines = funcy.lmap(str.strip, f.readlines())

        lyrics = "\n".join(lines)
        lyrics_tokenized = self.tokenizer.encode(lyrics)

        context = []
        target = []
        for i in range(len(lyrics_tokenized) - self.context_length):
            context_tokens = lyrics_tokenized[i:i + self.context_length]
            context.append(context_tokens)

            target_token = lyrics_tokenized[i + self.context_length]
            target.append(target_token)

        context = torch.stack(context)
        target = torch.tensor(target, dtype=torch.long)

        metadata = dict(
            artist=artist.title(),
            song_title=song_title.title(),
            filepath=filepath,
        )

        return context, target, metadata


if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(basedir, "data")

    dataset = LyricsDataset(datadir, context_length=8)

    context, target, metadata = dataset[0]

    artist = metadata["artist"]
    song_title = metadata["song_title"]
    filepath = metadata["filepath"]

    for context_tokens, target_token in zip(context, target):
        # for context, target in zip(context_tokens, target_token):
        target_token = target_token.unsqueeze(dim=0)
        print(f"""
{dataset.tokenizer.decode(context_tokens)}  -  {dataset.tokenizer.decode(target_token)}
""".strip())
