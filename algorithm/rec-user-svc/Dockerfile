FROM python:3.11-slim
RUN pip install --no-cache-dir fastapi uvicorn[standard] faiss-cpu  torch==2.7.0 redis
COPY app /app
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
