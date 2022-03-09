FROM python:3.8-slim
ENV PORT=8000
COPY requirements.txt /
RUN pip install -r requirements.txt
COPY ./app /app
COPY best_local_refined.h5 /
COPY breeds.txt /

ENTRYPOINT uvicorn app.main:app --host 0.0.0.0 --port $PORT
