FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN rm -rf mlflow.db mlruns && python ml/train.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
