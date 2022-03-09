from app.predictor import Predictor
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel


class PredictionSchema(BaseModel):
    filename: str
    content_type: str
    prediction: str
    top: dict


app = FastAPI(title="Dog breed predictor")
predictor = Predictor("breeds.txt", "best_local_refined.h5", 224)


@app.get("/")
def get_model_summary():
    """Returns a string with the model summary"""
    string_list = []
    predictor.model.summary(print_fn=lambda x: string_list.append(x))
    return {'summary': string_list}


@app.post("/predict", response_model=PredictionSchema)
def predict_image(n_top: int = 3, file: UploadFile = File(...)):
    """Receives a file and make a prediction on it"""
    prediction = predictor.predict_file(file.file, n_top=n_top)

    return PredictionSchema(
        filename=file.filename,
        content_type=file.content_type,
        prediction=prediction["label"],
        top=prediction["top"]
    )
