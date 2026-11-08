.PHONY: install etl train analyze test serve docker mlflow
install:  ; pip install -r requirements.txt
etl:      ; python -m src.etl_spark
train:    ; python -m src.train
analyze:  ; python -m src.analyze
test:     ; pytest -q
serve:    ; uvicorn app.main:app --reload
docker:   ; docker build -t churn-api . && docker run -p 8000:8000 churn-api
mlflow:   ; mlflow ui --backend-store-uri sqlite:///mlflow.db
