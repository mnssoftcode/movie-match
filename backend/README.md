# MovieMatch AI Backend

This is the backend service for MovieMatch that provides AI-powered movie recommendations based on user mood.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- FastAPI
- Uvicorn

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Make sure you have the trained model files in the `movie_model_tiny` folder.

## Running the Server

Start the server with:
```bash
uvicorn server:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## API Endpoints

- `GET /` - Health check endpoint
- `POST /predict` - Get genre predictions based on mood text

### Predict Endpoint

**Request:**
```json
{
  "text": "I'm feeling happy and excited today"
}
```

**Response:**
```json
{
  "genres": [
    ["Comedy", 0.95],
    ["Action", 0.87],
    ["Adventure", 0.78],
    ["Romance", 0.65],
    ["Family", 0.54]
  ]
}
```