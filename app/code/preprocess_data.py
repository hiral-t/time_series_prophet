import os
from app.utils.decorator import log_exception
from fastapi import APIRouter
from fastapi import File, UploadFile
from dotenv import load_dotenv
from app.service.preprocess_data_service import read_uploaded_csv, save_data_as_artifact, set_up_mlflow, preprocess_dataset, \
    split_dataset, run_adfuller_test, determine_seasonal_model, prepare_training_data


# load environment
load_dotenv()

# Load environment variables
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

router = APIRouter()

@log_exception
@router.post("/process-data/")
async def process_data(file: UploadFile = File(...)):
    """
        Process the uploaded CSV file, log parameters, and save the data.
        Args:
        - file: UploadFile object representing the uploaded CSV file.
        Returns:
        - A message indicating successful processing and parameter logging.
    """
    try:
        set_up_mlflow(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)
        retail_sales = await read_uploaded_csv(file)

        # Extract filename
        file_name = file.filename
        save_data_as_artifact(retail_sales, file_name)

        # Preprocess the dataset
        retail_sales = preprocess_dataset(retail_sales)

        # Split the dataset
        retail_sales_train, retail_sales_test, retail_sales_validate = split_dataset(retail_sales)

        # ad fuller test
        adf_results = run_adfuller_test(retail_sales)

        seasonal_model = determine_seasonal_model(retail_sales_train)
        print("Recommended seasonal model:", seasonal_model)

        prepare_training_data(retail_sales_train)

        return {"message": "Data processed and parameters logged successfully."}

    except Exception as e:
        return e

