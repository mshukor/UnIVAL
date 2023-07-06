from data.audio_utils import get_audio_features, int16_to_float32, float32_to_int16, AUDIO_CFG
import soundfile as sf
import io 
import torch 
import numpy as np

AUDIO_CFG =  {
        "audio_length": 1024,
        "clip_samples": 480000,
        "mel_bins": 64,
        "sample_rate": 48000,
        "window_size": 1024,
        "hop_size": 480,
        "fmin": 50,
        "fmax": 14000,
        "class_num": 527,
    }
    


audio_cfg = AUDIO_CFG
max_len = 480000
data_path = '/work/NAT/gda2204/mshukor/data/audiocaps/train/--CHY2qO5zc.wav'

audio_data, orig_sr = sf.read(data_path)
# import librosa
# audio_data, orig_sr = librosa.load(data_path, sr=48000)

print(orig_sr)
if audio_data.ndim>1:
  audio_data = np.mean(audio_data,axis=1)

 
print(audio_data.shape, audio_data)

audio_data = int16_to_float32(float32_to_int16(audio_data))
audio_data = torch.tensor(audio_data).float()
print(audio_data.dtype)
print(audio_data.shape, audio_data)
# the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
sample = {}

sample = get_audio_features(
    sample, audio_data, max_len, 
    data_truncating='fusion', 
    data_filling='repeatpad',
    audio_cfg=audio_cfg,
)

patch_audio = sample['waveform'] #.half()
print(patch_audio.shape, patch_audio.min(), patch_audio.max(), patch_audio)

patch_audio = torch.zeros(480000)
print(patch_audio.shape)


from torchlibrosa.stft import Spectrogram, LogmelFilterBank

AUDIO_CFG =  {
        "sample_rate": 48000,
        "audio_length": 1024,
        "clip_samples": 480000,
        "mel_bins": 64,
        "sample_rate": 48000,
        "window_size": 1024,
        "hop_size": 480,
        "fmin": 50,
        "fmax": 14000,
        "class_num": 527,
    }

window = 'hann'
center = True
pad_mode = 'reflect'
ref = 1.0
amin = 1e-10
top_db = None

spectrogram_extractor = Spectrogram(n_fft=AUDIO_CFG['window_size'], hop_length=AUDIO_CFG['hop_size'],
            win_length=AUDIO_CFG['window_size'], window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)


logmel_extractor = LogmelFilterBank(sr=AUDIO_CFG['sample_rate'], n_fft=AUDIO_CFG['window_size'],
            n_mels=AUDIO_CFG['mel_bins'], fmin=AUDIO_CFG['fmin'], fmax=AUDIO_CFG['fmax'], 
            ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)#.half()


patch_audio = patch_audio[None, :]
print(patch_audio.shape)
spectro = spectrogram_extractor(patch_audio)

print(spectro.shape)
print(spectro)


mel = logmel_extractor(spectro)

print(mel.shape)
print(mel)