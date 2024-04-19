from fastapi import FastAPI

from app.code.preprocess_data import router
from app.code.train_model import training_router
from app.code.inference import inference_router

def initialize_app():
    app = FastAPI()

    # configure router
    app.include_router(router)
    app.include_router(training_router)
    app.include_router(inference_router)

    # configure middleware

    return app


fastapi_app = initialize_app()
