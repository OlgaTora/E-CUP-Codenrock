FROM python:3.10-slim
WORKDIR /app
VOLUME /app/data
SHELL [ "/bin/bash", "-c" ]
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3","/app/make_submission.py"]

COPY . /app

RUN python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import tensorflow as tf; from tensorflow.keras.applications import EfficientNetB7; model = EfficientNetB7(weights='imagenet', include_top=False); model.save('/app/transfer_model.h5')" && \
    chmod +x /app/entrypoint.sh /app/baseline.py /app/make_submission.py


