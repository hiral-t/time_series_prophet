# time_series_prophet

Sales Data Model Fine-tuning and Inference with MLflow
This repository contains code to fine-tune a sales data model, perform inference on new data, and store the results in MLflow. The process involves loading the pre-trained model, fine-tuning it with sales data, generating forecasts, and logging the results in MLflow.

## Prerequisites

- Python 3.x installed
- MLflow installed (`pip install mlflow`)
- Other dependencies installed (`pip install -r requirements.txt`)


## Usage

1. **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
## APIs

This repository provides the following APIs:

1. **Preprocess Data API:**

    Endpoint: `/process_data/`

    Description: Preprocesses sales data to make it compatible with the model.

2. **Fine-tune Model API:**

    Endpoint: `/train-model/`

    Description: Train the pre-trained model with sales data.

3. **Perform Inference API:**

    Endpoint: `/predict-forecast/`

    Description: Performs inference on new sales data using the trained model.