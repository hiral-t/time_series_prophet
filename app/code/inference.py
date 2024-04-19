
from http.client import HTTPException
from fastapi import  Response, HTTPException

from app.service.inference_service import predict_and_plot_forecast
from app.utils.decorator import log_exception
from fastapi import APIRouter

inference_router = APIRouter()


@log_exception
@inference_router.get("/predict-forecast/")
def get_forecast(num_months: int):
    """
    Get forecast for the specified number of months.
    """
    try:
        if num_months <= 0:
            raise HTTPException(status_code=400, detail="Number of months must be positive.")
        img_buf = predict_and_plot_forecast(num_months)
        return Response(content=img_buf.getvalue(), media_type="image/png")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
