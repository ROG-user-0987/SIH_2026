FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /srv

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" transformers huggingface_hub safetensors

COPY server.py .
COPY spectra_aasist3_model.py .

ENV MODEL_NAME=spectra
ENV HOST=0.0.0.0
ENV PORT=8001

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]