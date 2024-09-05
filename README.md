# Stock Price Prediction

This project is a stock price prediction tool that uses a Long Short-Term Memory (LSTM) model. The backend is built with Flask and TensorFlow, while the frontend is developed using Angular. The tool predicts future stock prices based on historical data.

## Features

- Fetch historical stock data from external APIs.
- Train an LSTM model for stock price prediction.
- Display predictions and model accuracy on a user-friendly web interface.

## Tech Stack
- Backend developed using Flask, Tensorflow, and Python.
- Frontend created with Angular and Typescript.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Node.js and npm
- Angular CLI

### Backend Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/jwmason/Stock-Price-Predictor.git
    ```

2. **Navigate to the backend**:

    ```bash
    cd backend
    ```

3. **Run The App**:

    ```bash
    python app.py
    ```
The backend will be running on [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Frontend Setup

1. **Navigate to the frontend**:

    ```bash
    cd frontend
    ```

2. **Install dependencies**:

    ```bash
    npm install
    ```

3. **Run The App**:

    ```bash
    ng serve
    ```
The frontend will be running on [http://localhost:4200/](http://localhost:4200/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Yahoo Finance for data fetching.
- Tensorflow for the LSTM neural network model.
- Angular and Flask communities for their contributions and support.
