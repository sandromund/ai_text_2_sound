FROM python:3.9-slim
COPY ./app /app
COPY . /app
RUN pip3 install  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r app/requirements.txt
RUN pip install transformers
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "15400" ]