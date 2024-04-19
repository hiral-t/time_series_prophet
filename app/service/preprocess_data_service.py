import io

import joblib
from fastapi import UploadFile
import pandas as pd
import mlflow
from pandas._libs.tslibs.offsets import MonthEnd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from fbprophet import Prophet

from app.utils import constant


def set_up_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """
    Set up MLflow for tracking experiments.

    Args:
    - tracking_uri: URI of the MLflow tracking server.
    - experiment_name: Name of the MLflow experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print("Experiment created")
        mlflow.create_experiment(experiment_name, "mlflow-artifacts:/0")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()


async def read_uploaded_csv(file: UploadFile) -> pd.DataFrame:
    """
    Read the uploaded CSV file into a pandas DataFrame.
    Args:
    - file: UploadFile object representing the uploaded CSV file.
    Returns:
    - Pandas DataFrame containing the data from the CSV file.
    """
    content = await file.read()
    retail_sales = pd.read_csv(io.BytesIO(content), index_col=0)
    return retail_sales


def save_data_as_artifact(retail_sales: pd.DataFrame, file_name: str) -> None:
    """
    Save the data as an artifact in MLflow.
    Args:
    - retail_sales: Pandas DataFrame containing the data.
    - file_name: Name of the uploaded file.
    """
    retail_sales.to_csv(constant.RAW_DATA_FILE_PATH + file_name)

    # Create an instance of a PandasDataset
    dataset = mlflow.data.from_pandas(
        retail_sales, source=constant.RAW_DATA_FILE_PATH + file_name
    )
    # # Log the Dataset to an MLflow run by using the `log_input` API
    mlflow.log_input(dataset, context="training")

    mlflow.log_artifact(constant.RAW_DATA_FILE_PATH + file_name, artifact_path="datasets")


def preprocess_dataset(retail_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by converting start dates to end dates.
    Args:
    - retail_sales: Pandas DataFrame containing the data.
    Returns:
    - Preprocessed Pandas DataFrame with end dates.
    """
    retail_sales.index = pd.to_datetime(retail_sales.index)
    retail_sales.reset_index(inplace=True)
    retail_sales['month'] = pd.to_datetime(retail_sales['month']) + MonthEnd(1)
    retail_sales.set_index(['month'], inplace=True)
    retail_sales.index = pd.to_datetime(retail_sales.index)
    return retail_sales


def split_dataset(retail_sales: pd.DataFrame) -> tuple:
    """
    Split the dataset into train, test, and validation sets.
    Args:
    - retail_sales: Pandas DataFrame containing the data.
    Returns:
    - Tuple containing train, test, and validation sets.
    """
    retail_sales_actuals = retail_sales.loc['2005-01-01':'2018-12-31']
    retail_sales_train = retail_sales_actuals.loc['2005-01-01':'2016-12-31']
    retail_sales_test = retail_sales_actuals.loc['2017-01-01':'2018-12-31']
    retail_sales_traintest = retail_sales_actuals.loc['2005-01-01':'2018-12-31']
    retail_sales_validate = retail_sales.loc['2019-01-01':'2021-12-31']
    retail_sales_cross_validate = retail_sales.loc['2017-01-01':'2021-12-31']

    mlflow.log_param("Actuals", retail_sales_actuals.shape)
    mlflow.log_param("Train", retail_sales_train.shape)
    mlflow.log_param("Test", retail_sales_test.shape)
    mlflow.log_param("Train and Test", retail_sales_traintest.shape)
    mlflow.log_param("Validate", retail_sales_validate.shape)
    mlflow.log_param("Cross Validate", retail_sales_cross_validate.shape)

    return retail_sales_train, retail_sales_test, retail_sales_validate


def run_adfuller_test(retail_sales: pd.DataFrame):
    """
    Run the Augmented Dickey-Fuller test on the sales_total column.
    Args:
    - retail_sales: Pandas DataFrame containing the data.
    Returns:
    - Tuple containing the results of the ADF test.
    """
    adf, pval, usedlag, nobs, crit_vals, icbest = adfuller(retail_sales.sales_total.values)

    print('ADF test statistic:', adf)
    print('ADF p-values:', pval)
    print('ADF number of lags used:', usedlag)
    print('ADF number of observations:', nobs)
    print('ADF critical values:', crit_vals)
    print('ADF best information criterion:', icbest)

    mlflow.log_param("ADF_test_statistic", adf)
    mlflow.log_param("ADF_p_values", pval)
    mlflow.log_param("ADF_number_of_lags_used", usedlag)
    mlflow.log_param("ADF_number_of_observations", nobs)
    mlflow.log_param("ADF_critical_values", crit_vals)
    mlflow.log_param("ADF_best_information_criterion", icbest)
    return True


def determine_seasonal_model(retail_sales_train: pd.DataFrame) -> str:
    """
    Determine whether to use the multiplicative or additive model for seasonal decomposition.
    Args:
    - retail_sales_train: Pandas DataFrame containing the training data.
    Returns:
    - String indicating the recommended seasonal model ('multiplicative' or 'additive').
    """
    retail_sales_train.index = pd.to_datetime(retail_sales_train.index)

    # Perform seasonal decomposition using multiplicative model
    retail_sales_train_decompose_multi = seasonal_decompose(retail_sales_train, model='multiplicative')
    retail_sales_train_decompose_multi_resid = retail_sales_train_decompose_multi.resid.sum()

    # Perform seasonal decomposition using additive model
    retail_sales_train_decompose_add = seasonal_decompose(retail_sales_train, model='additive')
    retail_sales_train_decompose_add_resid = retail_sales_train_decompose_add.resid.sum()

    # Determine which model to use based on the sum of residuals
    if retail_sales_train_decompose_multi_resid < retail_sales_train_decompose_add_resid:
        return "multiplicative"
    else:
        return "additive"


def prepare_training_data(retail_sales_train: pd.DataFrame) -> None:
    """
    Prepare the training data by resetting the index, renaming columns, and saving it to a CSV file.
    Args:
    - retail_sales_train: Pandas DataFrame containing the training data.
    - output_file: Name of the CSV file to save the training data.
    """
    # Reset index and rename columns
    retail_sales_train = retail_sales_train.reset_index()
    retail_sales_train.columns = ['ds', 'y']

    # Save the training data to a CSV file
    retail_sales_train.to_csv(constant.TRAINING_DATA_PATH + constant.TRAINING_FILE_NAME, index=False)

    print("Training data saved to:", constant.TRAINING_DATA_PATH + constant.TRAINING_FILE_NAME)


def read_training_data(file) -> pd.DataFrame:
    """
    Read the uploaded CSV file containing training data into a pandas DataFrame.
    Args:
    - file: UploadFile object representing the uploaded CSV file.
    Returns:
    - Pandas DataFrame containing the training data.
    """
    # content = file.read()
    retail_sales_train = pd.read_csv(file)
    return retail_sales_train


def train_prophet_model(retail_sales_train: pd.DataFrame):
    """
    Train a Prophet model using the provided training data.
    Args:
    - retail_sales_train: Pandas DataFrame containing the training data.
    Returns:
    - Trained Prophet model.
    """
    retail_sales_model = Prophet(yearly_seasonality=True, seasonality_mode='multiplicative')
    retail_sales_model.fit(retail_sales_train)
    return retail_sales_model
