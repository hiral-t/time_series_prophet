import io
from http.client import HTTPException
from fastapi import  Response, HTTPException

import joblib
import matplotlib.pyplot as plt
from app.utils import constant


def predict_and_plot_forecast(num_months):
    """
    Generate forecast for the specified number of months, plot it, and save the plot as an image.
    """
    try:
        # Load the model from the given path
        retail_sales_model = joblib.load(constant.TRAINED_MODEL_PATH)

        retail_sales_future = retail_sales_model.make_future_dataframe(freq='M', periods=num_months)
        retail_sales_forecast = retail_sales_model.predict(retail_sales_future)

        # Plot forecast
        fig = retail_sales_model.plot(retail_sales_forecast)
        # fig.savefig("forecast_plot.png")

        # Convert the plot to a PNG image
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.seek(0)
        plt.close(fig)

        # Return the image as a FastAPI response
        return img_buf

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))