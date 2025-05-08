# Mini-Rec Algorithm Testing Guide

This guide explains how to run tests and debug the mini-recommendation system.

## Project Structure

The project consists of three microservices:
- **gateway-svc**: Entry point that coordinates the recommendation flow
- **rec-user-svc**: Handles item recall using FAISS similarity search
- **ranker-svc**: Scores and ranks candidate items

## Prerequisites

Make sure you have the following dependencies installed:
- Python 3.7+
- PyTorch
- FAISS
- Redis
- FastAPI
- httpx
- NumPy

## Running Tests

### Option 1: Run All Tests

```bash
python tests/run_tests.py
```

### Option 2: Run Individual Tests

```bash
# Test ranking service
python -m unittest tests.test_rank

# Test recall service
python -m unittest tests.test_recall

# Test gateway service
python -m unittest tests.test_gateway
```

## Debugging Tips

### Common Issues and Solutions

1. **Redis Connection Issues**
   - Ensure Redis server is running
   - Check the Redis URL in settings

2. **Missing Model Files**
   - Verify that model files exist in the expected locations
   - Check paths in settings.py configuration

3. **FAISS Index Problems**
   - Ensure FAISS index is properly built and available
   - Verify vector dimensions match between user and item embeddings

4. **Network Connectivity Between Services**
   - Check if services can communicate with each other
   - Verify service URLs in gateway-svc configuration

## Manual Testing

You can manually test the recommendation flow with curl:

```bash
# Get recommendations for user 123
curl "http://localhost:8000/recommend/123?k=10"

# Create a user profile
curl -X POST "http://localhost:8000/profile" \
  -H "Content-Type: application/json" \
  -d '{"userId": "test_user", "ratings": [{"movieId": 1, "score": 5.0}, {"movieId": 2, "score": 4.0}]}'
```

## Integration Testing with Docker

The full system can be tested using docker-compose:

```bash
# Start all services
docker-compose up -d

# Run integration tests
python integration_tests.py

# Shutdown services
docker-compose down
```
