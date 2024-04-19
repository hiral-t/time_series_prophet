from app.service.preprocess_data_service import read_training_data, train_prophet_model
from app.service.train_model_service import save_model

from app.utils import constant
from app.utils.decorator import log_exception
from fastapi import APIRouter

training_router = APIRouter()


@log_exception
@training_router.post("/train-model/")
def train_model():
    """
    Returns:
    - A message indicating successful processing and parameter logging.
    """
    try:
        # Read training data
        retail_sales_train = read_training_data(constant.TRAINING_DATA_PATH + constant.TRAINING_FILE_NAME)

        retail_sales_model = train_prophet_model(retail_sales_train)

        # Save the trained model to an artifacts file
        save_model(retail_sales_model, constant.TRAINED_MODEL_PATH)

        return {"message": "model trained successfully."}
    except Exception as e:
        return e

