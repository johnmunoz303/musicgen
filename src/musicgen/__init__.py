import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID="facebook/musicgen-small"
DEVICE="cpu"

PROMPT="128 BPM, electronic dance beat, energetic synthwave lead, pulsing analog bassline, four-on-the-floor kick drum, crisp hi-hats, futuristic atmosphere, clean mix, instrumental"

#torch.set_num_threads(4)

print("Cargando MusicGen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()

print("Preparando el prompt")

inputs = processor(
                  text=[PROMPT],
                  padding=True,
                  return_tensors="pt"
              ).to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio=model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=200 # 4s 750 ~ 15s
    )

sample_rate=model.config.audio_encoder.sampling_rate # ~32000 Hz
waveform = audio[0, 0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Musica generada")
