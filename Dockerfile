FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

CMD ["python", "src/predict.py", "This project is awesome!", "--model", "models/sentiment.joblib"]