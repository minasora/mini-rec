FROM pytorch/pytorch:2.7.0-cpu
RUN pip install --no-cache-dir fastapi uvicorn[standard] faiss-cpu  torch==2.7.0 redis
COPY app /app
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
