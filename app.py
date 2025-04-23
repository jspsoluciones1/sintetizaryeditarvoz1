import gradio as gr
import torch
import soundfile as sf
import os
from f5_tts import F5TTS

# Inicializar el modelo
tts = F5TTS()

def generar_audio(ref_audio, ref_texto, gen_texto, pitch, velocidad, brillo, volumen, reverb, eco, normalizar):
    """Genera audio con los efectos seleccionados"""
    # Crear directorio de salida si no existe
    if not os.path.exists("salida"):
        os.makedirs("salida")
    
    # Configurar efectos de audio
    audio_effects = [{
        'pitch_shift': pitch,
        'speed': velocidad,
        'brightness': brillo,
        'volume': volumen,
        'reverb': reverb,
        'echo': eco,
        'normalize': normalizar,
        'equalizer': {
            '60Hz': 0,
            '170Hz': 0,
            '310Hz': 0,
            '600Hz': 0,
            '1kHz': 0,
            '3kHz': 0,
            '6kHz': 0,
            '12kHz': 0,
            '14kHz': 0,
            '16kHz': 0
        }
    }]
    
    # Generar audio
    wav, sr, _ = tts.infer(
        ref_file=ref_audio,
        ref_text=ref_texto,
        gen_text=gen_texto,
        audio_effects=audio_effects
    )
    
    # Guardar audio
    output_file = "salida/audio_generado.wav"
    sf.write(output_file, wav, sr)
    
    return output_file

# Crear interfaz de Gradio
with gr.Blocks(title="F5-TTS con Efectos de Audio") as demo:
    gr.Markdown("# 🎙️ F5-TTS con Efectos de Audio")
    gr.Markdown("Genera audio con diferentes efectos de sonido")
    
    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(label="Audio de Referencia", type="filepath")
            ref_texto = gr.Textbox(label="Texto de Referencia", placeholder="Ingresa el texto de referencia...")
            gen_texto = gr.Textbox(label="Texto a Generar", placeholder="Ingresa el texto que quieres generar...")
        
        with gr.Column():
            pitch = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="Tono (semitones)")
            velocidad = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Velocidad")
            brillo = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Brillo")
            volumen = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Volumen")
            reverb = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Reverberación")
            eco = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Eco")
            normalizar = gr.Checkbox(label="Normalizar Audio", value=True)
    
    generar_btn = gr.Button("🎵 Generar Audio")
    output_audio = gr.Audio(label="Audio Generado", type="filepath")
    
    generar_btn.click(
        fn=generar_audio,
        inputs=[ref_audio, ref_texto, gen_texto, pitch, velocidad, brillo, volumen, reverb, eco, normalizar],
        outputs=output_audio
    )

# Para ejecutar localmente
if __name__ == "__main__":
    demo.launch() 
