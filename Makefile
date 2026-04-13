.PHONY: install train predict serve test docker-build docker-up docker-down azure-deploy

install:
	pip install -r requirements.txt

train:
	python pipelines/train_pipeline.py --config configs/config.yaml

predict:
	python pipelines/predict_pipeline.py \
		--input data/raw/Fertility_Health_Dataset_2026.csv \
		--output data/processed/predictions.csv

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

docker-build:
	docker build -f docker/Dockerfile -t fertility-outcome-api:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down

azure-deploy:
	az acr login --name fertilitymlacr
	docker tag fertility-outcome-api:latest fertilitymlacr.azurecr.io/fertility-outcome-api:latest
	docker push fertilitymlacr.azurecr.io/fertility-outcome-api:latest
	az containerapp update \
		--name fertility-outcome-api \
		--resource-group rg-fertility-ml \
		--image fertilitymlacr.azurecr.io/fertility-outcome-api:latest
