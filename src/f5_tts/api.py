import random
import sys
from importlib.resources import files

import soundfile as sf
import torch
import tqdm
from cached_path import cached_path

from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    target_sample_rate,
)
from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything
from f5_tts.audio_editor import AudioEditor


class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_name="vocos",
        local_path=None,
        device=None,
    ):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name
        self.audio_editor = AudioEditor(sample_rate=target_sample_rate)

        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load models
        self.load_vocoder_model(vocoder_name, local_path)
        self.load_ema_model(model_type, ckpt_file, vocoder_name, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, vocoder_name, local_path):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema):
        if model_type == "F5-TTS":
            if not ckpt_file:
                if mel_spec_type == "vocos":
                    ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
                elif mel_spec_type == "bigvgan":
                    ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt"))
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device
        )

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
        save_spectrogram(spect, file_spect)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
        audio_effects=None,
    ):
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, device=self.device)

        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        # Apply audio effects if specified
        if audio_effects is not None:
            wav = self.audio_editor.apply_effects(wav, audio_effects)

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spect is not None:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect


if __name__ == "__main__":
    # Ejemplo de uso básico
    print("Inicializando F5-TTS...")
    f5tts = F5TTS()

    # Ejemplo 1: Voz clara y brillante
    print("\nEjemplo 1: Voz clara y brillante")
    audio_effects_1 = [{
        'equalizer': {
            '3kHz': 3,
            '6kHz': 4,
            '12kHz': 3
        },
        'compressor': {
            'threshold': -18,
            'ratio': 3
        }
    }]

    # Ejemplo 2: Voz grave y cálida
    print("\nEjemplo 2: Voz grave y cálida")
    audio_effects_2 = [{
        'equalizer': {
            '60Hz': 4,
            '170Hz': 3,
            '310Hz': 2
        },
        'reverb': 0.2
    }]

    # Ejemplo 3: Voz natural y suave
    print("\nEjemplo 3: Voz natural y suave")
    audio_effects_3 = [{
        'equalizer': {
            '600Hz': 2,
            '1kHz': 1,
            '3kHz': 1
        },
        'compressor': {
            'threshold': -20,
            'ratio': 2,
            'attack': 30,
            'release': 150
        }
    }]

    # Ejemplo 4: Voz con eco y reverberación
    print("\nEjemplo 4: Voz con eco y reverberación")
    audio_effects_4 = [{
        'reverb': 0.3,
        'echo': 0.2,
        'equalizer': {
            '1kHz': 2,
            '3kHz': 1
        }
    }]

    # Ejemplo 5: Voz con todos los efectos
    print("\nEjemplo 5: Voz con todos los efectos")
    audio_effects_5 = [{
        'equalizer': {
            '60Hz': 2,
            '170Hz': 1,
            '310Hz': 1,
            '600Hz': 1,
            '1kHz': 2,
            '3kHz': 2,
            '6kHz': 1,
            '12kHz': 1
        },
        'reverb': 0.2,
        'echo': 0.1,
        'compressor': {
            'threshold': -20,
            'ratio': 3,
            'attack': 20,
            'release': 100
        },
        'speed': 1.1,
        'volume': 1.2,
        'normalize': True
    }]

    # Generar audio con efectos
    print("\nGenerando audio con efectos...")
    wav, sr, spect = f5tts.infer(
        ref_file="referencia.wav",
        ref_text="texto de referencia",
        gen_text="texto a generar",
        audio_effects=audio_effects_1  # Puedes cambiar a audio_effects_2, 3, 4 o 5
    )
    print("¡Audio generado exitosamente!")

