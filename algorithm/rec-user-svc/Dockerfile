FROM minasora693/pytorch-fastapi:latest
RUN pip install --no-cache-dir faiss-cpu  pandas
COPY app /app
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
