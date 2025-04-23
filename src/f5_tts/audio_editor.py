import numpy as np
import torch
import torchaudio
import ffmpeg
import tempfile
import os
import subprocess
from pydub import AudioSegment
from pydub.effects import speedup, pitch_shift, normalize

class AudioEditor:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        """Verifica que FFmpeg esté instalado en el sistema"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg no está instalado. Por favor, instala FFmpeg en tu sistema:\n"
                "Windows: https://ffmpeg.org/download.html\n"
                "Linux: sudo apt-get install ffmpeg\n"
                "macOS: brew install ffmpeg"
            )

    def edit_audio(self, audio_data, params):
        """
        Edita el audio con varios parámetros
        
        Args:
            audio_data: array numpy o tensor torch de datos de audio
            params: diccionario con parámetros de edición:
                - pitch_shift: float, semitonos para cambiar el tono (default: 0)
                - speed: float, multiplicador de velocidad (default: 1.0)
                - brightness: float, ajuste de brillo (default: 1.0)
                - volume: float, multiplicador de volumen (default: 1.0)
                - normalize: bool, normalizar el audio (default: False)
                - remove_silence: bool, eliminar silencios (default: False)
                - silence_threshold: float, umbral de silencio en dB (default: -50)
                - equalizer: dict, configuración del ecualizador (default: None)
                - reverb: float, cantidad de reverberación (default: 0.0)
                - echo: float, cantidad de eco (default: 0.0)
                - compressor: dict, configuración del compresor (default: None)
        
        Returns:
            edited_audio: array numpy del audio editado
        """
        # Convertir a numpy si es un tensor torch
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        
        # Convertir a AudioSegment para procesamiento
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1
        )
        
        # Aplicar cambio de tono si se especifica
        if 'pitch_shift' in params and params['pitch_shift'] != 0:
            audio_segment = pitch_shift(audio_segment, params['pitch_shift'])
        
        # Aplicar ajuste de velocidad si se especifica
        if 'speed' in params and params['speed'] != 1.0:
            audio_segment = speedup(audio_segment, params['speed'])
        
        # Aplicar ajuste de brillo (usando ecualización)
        if 'brightness' in params and params['brightness'] != 1.0:
            brightness_db = 20 * np.log10(params['brightness'])
            audio_segment = audio_segment.apply_gain(brightness_db)
        
        # Aplicar ajuste de volumen
        if 'volume' in params and params['volume'] != 1.0:
            volume_db = 20 * np.log10(params['volume'])
            audio_segment = audio_segment.apply_gain(volume_db)
        
        # Normalizar si se solicita
        if params.get('normalize', False):
            audio_segment = normalize(audio_segment)
        
        # Eliminar silencios si se solicita
        if params.get('remove_silence', False):
            silence_threshold = params.get('silence_threshold', -50)
            audio_segment = self._remove_silence(audio_segment, silence_threshold)
        
        # Aplicar ecualizador si se especifica
        if 'equalizer' in params and params['equalizer'] is not None:
            audio_segment = self._apply_equalizer(audio_segment, params['equalizer'])
        
        # Aplicar efectos de FFmpeg si se especifican
        if any(key in params for key in ['reverb', 'echo', 'compressor']):
            audio_segment = self._apply_ffmpeg_effects(audio_segment, params)
        
        # Convertir de vuelta a array numpy
        edited_audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        edited_audio = edited_audio / 32768.0  # Normalizar a [-1, 1]
        
        return edited_audio

    def _remove_silence(self, audio_segment, silence_threshold=-50):
        """Elimina silencios del audio usando la detección de silencio de pydub"""
        from pydub.silence import split_on_silence
        
        # Dividir audio en silencios
        audio_chunks = split_on_silence(
            audio_segment,
            min_silence_len=100,
            silence_thresh=silence_threshold,
            keep_silence=100
        )
        
        # Combinar chunks no silenciosos
        if audio_chunks:
            return sum(audio_chunks)
        return audio_segment

    def _apply_equalizer(self, audio_segment, eq_params):
        """
        Aplica ecualización al audio
        
        Args:
            audio_segment: AudioSegment a ecualizar
            eq_params: diccionario con bandas de ecualización
        """
        try:
            # Crear archivo temporal para el audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_in:
                audio_segment.export(temp_in.name, format='wav')
                
                # Construir filtro de ecualización
                eq_filter = []
                for freq, gain in eq_params.items():
                    if gain != 0:
                        eq_filter.append(f"equalizer=frequency={freq}:width_type=octave:width=1:gain={gain}")
                
                if eq_filter:
                    try:
                        # Aplicar ecualización con FFmpeg
                        stream = ffmpeg.input(temp_in.name)
                        stream = ffmpeg.filter(stream, ','.join(eq_filter))
                        
                        # Guardar resultado en archivo temporal
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
                            stream.output(temp_out.name).overwrite_output().run(capture_stdout=True, capture_stderr=True)
                            
                            # Cargar resultado
                            result = AudioSegment.from_wav(temp_out.name)
                            os.unlink(temp_out.name)
                    except ffmpeg.Error as e:
                        print(f"Error en FFmpeg durante la ecualización: {e.stderr.decode()}")
                        return audio_segment
                else:
                    result = audio_segment
                
                os.unlink(temp_in.name)
                return result
        except Exception as e:
            print(f"Error durante la ecualización: {str(e)}")
            return audio_segment

    def _apply_ffmpeg_effects(self, audio_segment, params):
        """
        Aplica efectos de FFmpeg al audio
        
        Args:
            audio_segment: AudioSegment a procesar
            params: diccionario con parámetros de efectos
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_in:
                audio_segment.export(temp_in.name, format='wav')
                
                # Construir filtros de FFmpeg
                filters = []
                
                # Aplicar reverberación
                if 'reverb' in params and params['reverb'] > 0:
                    reverb_amount = params['reverb']
                    filters.append(f"aecho=0.8:0.9:{int(1000*reverb_amount)}:0.3")
                
                # Aplicar eco
                if 'echo' in params and params['echo'] > 0:
                    echo_amount = params['echo']
                    filters.append(f"aecho=0.8:0.9:{int(1000*echo_amount)}:0.3")
                
                # Aplicar compresor
                if 'compressor' in params and params['compressor'] is not None:
                    comp = params['compressor']
                    threshold = comp.get('threshold', -20)
                    ratio = comp.get('ratio', 4)
                    attack = comp.get('attack', 20)
                    release = comp.get('release', 100)
                    filters.append(f"acompressor=threshold={threshold}:ratio={ratio}:attack={attack}:release={release}")
                
                if filters:
                    try:
                        # Aplicar efectos con FFmpeg
                        stream = ffmpeg.input(temp_in.name)
                        stream = ffmpeg.filter(stream, ','.join(filters))
                        
                        # Guardar resultado en archivo temporal
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
                            stream.output(temp_out.name).overwrite_output().run(capture_stdout=True, capture_stderr=True)
                            
                            # Cargar resultado
                            result = AudioSegment.from_wav(temp_out.name)
                            os.unlink(temp_out.name)
                    except ffmpeg.Error as e:
                        print(f"Error en FFmpeg durante la aplicación de efectos: {e.stderr.decode()}")
                        return audio_segment
                else:
                    result = audio_segment
                
                os.unlink(temp_in.name)
                return result
        except Exception as e:
            print(f"Error durante la aplicación de efectos: {str(e)}")
            return audio_segment

    def apply_effects(self, audio_data, effects):
        """
        Aplica múltiples efectos al audio en secuencia
        
        Args:
            audio_data: array numpy o tensor torch de datos de audio
            effects: lista de diccionarios, cada uno con parámetros de efectos
        
        Returns:
            edited_audio: array numpy del audio editado
        """
        current_audio = audio_data
        for effect in effects:
            current_audio = self.edit_audio(current_audio, effect)
        return current_audio 
