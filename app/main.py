from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.utils import predict_v1, predict_v2

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...), model: str = Form(...)):

    if model == "v1":
        result = predict_v1(text)
    else:
        result = predict_v2(text)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )