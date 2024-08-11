# Stock Price Prediction

This project provides a stock price prediction tool using a Long Short-Term Memory (LSTM) model. The backend is built with Flask and TensorFlow, while the frontend is developed using Angular. The project predicts future stock prices based on historical data.

## Features

- Fetch historical stock data.
- Train an LSTM model to predict stock prices.
- Display predictions and accuracy on a web frontend.
- Built using Flask for the backend and Angular for the frontend.

## Getting Started

### Prerequisites

- Python 3.x
- Node.js and npm
- Angular CLI

### Setup Instructions

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd stock-price-predictor

#### 2. Backend Setup

cd backend
pip install -r requirements.txt
python app.py

#### 3. Frontend Setup

cd frontend
npm install
ng serve

### Project Structure

stock-price-predictor/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── (other backend files)
└── frontend/
    ├── src/
    ├── angular.json
    ├── package.json
    └── (other frontend files)

