from fastapi import FastAPI
from pydantic import BaseModel, Field

from transformers import AutoProcessor, BarkModel
import scipy
import warnings

voice_preset="v2/de_speaker_8"
warnings.filterwarnings("ignore")
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to("cuda")






class ModelInput(BaseModel):
    text: str

class ModelOutput(BaseModel):
    text : str
    sample_rate: int
    audio_array : list

app = FastAPI()



@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "1"}



@app.post("/audio", response_model=ModelOutput)
def generate_audio(request : ModelInput):
    inputs = processor(request["text"], voice_preset=voice_preset)
    for k, v in inputs.items():
        inputs[k] = v.to("cuda")
    audio_array = model.generate(**inputs, pad_token_id=10000)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    return {"text" : request["text"], 
            "sample_rate" : sample_rate, 
            "audio_array": audio_array.tolist()}