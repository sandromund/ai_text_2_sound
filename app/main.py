from fastapi import FastAPI
from pydantic import BaseModel, Field


class ModelInput(BaseModel):
    text: str

class ModelOutput(BaseModel):
    text: str

app = FastAPI()



@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "1"}


@app.post('/predict', response_model=ModelOutput)
async def model_predict(input: ModelInput):
    return input