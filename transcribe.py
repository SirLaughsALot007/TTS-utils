import argparse
import torch
import whisper
import os
import torchaudio
import json

def transcribe_one(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)
    print(result.text)
    return lang, result.text

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="CJ", choices=["C", "CJ", "CJE"])
    parser.add_argument("--whisper_size", default="large", choices=["small", "medium", "large"])
    parser.add_argument("--audio_dir", type=str, help="dir path of the audio files, it should contains one or some folders which are the speaker names")
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate of the audio files")
    args = parser.parse_args()

    if args.base_model == "CJE":
        lang2token = {
            'zh' : "[ZH]",
            'ja' : "[JA]",
            "en" : "[EN]",
        }
    elif args.base_model == "CJ":
        lang2token = {
            'zh' : "[ZH]",
            'ja' : "[JA]",
        }
    elif args.base_model == "C":
        lang2token = {
            'zh' : "[ZH]",
        }

    assert torch.cuda.is_available(), "Please enable GPU in order to run Whisper!"

    model = whisper.load_model(args.whisper_size)
    speaker_names = list(os.walk(args.audio_dir))[0][1] # 获取所有子目录的名字即speakername

    # 遍历所有speaker的文件夹
    for speaker_name in speaker_names:
        for i, wavfile in enumerate(list(os.walk(os.path.join(args.audio_dir, speaker_name)))):
            try:
                filepath = os.path.join(args.audio_dir, speaker_name, wavfile)
                wav, sr = torchaudio.load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
                wav = wav.mean(dim=0).unsqueeze(0) # 将多声道（多通道）的音频变换为单声道

                if sr != args.sr:
                   print('Resampling from %d to %d' % (sr, args.sr))
                   wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=args.sr)(wav)
                lang, text = transcribe_one(wav)
            except Exception as e:
                print(e)
                continue

            
            
                




    


    
