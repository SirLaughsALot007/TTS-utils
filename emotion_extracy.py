import torch
import torch.nn as nn
import os
import librosa


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import(
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()
    def forward(self, input_wav):
        outputs = self.wav2vec2(input_wav)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logit = self.classifier(hidden_states)

        return hidden_states, logit


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)

import numpy as np
def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    """
    Predict emotions or extract embeddings from raw audio signal.
    """
    y = processor(x, sampling_rate = sampling_rate)
    y = y['input_values'][0]
    y = np.expand_dims(y, 0)
    y = torch.from_numpy(y).to(device)
    with torch.no_grad():
        
        y = model(y)[0 if embeddings else 1]
    y = y.detach().cpu().numpy()
    return y  # [Arousal, dominance, valence]

embs = []
wavnames = []

def extract_dir(path):
    rootpath = path
    for idx, wavname in enumerate(os.listdir(rootpath)):
        wav, sr = librosa.load(os.path.join(rootpath, wavname), sr=16000)
        emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
        embs.append(emb)
        wavnames.append(wavname)
        np.save(os.path.join(rootpath, wavname + '.emo.npy'), emb.squeeze(0))
        print(idx, wavname)

def extract_wav(path):
    wav, sr = librosa.load(path, 16000)
    emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
    return emb

def preprocess_one(path):
    print(path)
    wav, sr = librosa.load(path, sr=16000)
    emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
    savepath = os.path.join(args.output_path, path.split('/')[-1].split('.')[0] + '.emo.npy')
    np.save(savepath, emb.squeeze(0))
    return emb

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Emotion Extraction Preprocess")
    parser.add_argument('--output_path', type=str, default='/home/sjx/Common/vits/VITS-fast-fine-tuning/emotion_embedding')
    parser.add_argument('--filedir', type=str, help='path of the filelists')
    global args
    args = parser.parse_args()
    filelists = os.listdir(args.filedir)
    for file in filelists:
        path = os.path.join(args.filedir, file)
        print(file, "------start emotion extract-------")
        preprocess_one(path)
