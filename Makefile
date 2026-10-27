PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup setup-dl demo train-seg train-progression evaluate report api web test test-backend test-frontend docs clean-data

setup:            ## Create the venv and install backend + frontend dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e "backend[dev]"
	npm --prefix frontend install

setup-dl:         ## Add PyTorch + MONAI for the deep segmentation model
	$(PIP) install -e "backend[dl]"

demo:             ## Generate the DEMO DATA cases (also done automatically at API start-up)
	cd backend && ../$(PY) -c "from onconova.config import get_settings; from onconova.store import CaseStore; from onconova.synthetic.cases import materialize_demo_cases; s=get_settings(); s.ensure_dirs(); print(materialize_demo_cases(CaseStore(s.cases_dir)))"

train-seg:        ## Generate the synthetic dataset (if needed) and train the demo 3-D U-Net
	cd backend && ../$(PY) scripts/train_segmentation.py

train-progression: ## Train and evaluate the demo progression model on the synthetic cohort
	cd backend && ../$(PY) scripts/train_progression.py

evaluate:         ## Evaluate every available segmentation model on the held-out test patients
	cd backend && ../$(PY) scripts/evaluate_segmentation.py

report:           ## Regenerate docs/EVALUATION_REPORT.md and docs/API.md from measured artifacts and the live API schema
	cd backend && ../$(PY) scripts/build_evaluation_report.py && ../$(PY) scripts/export_openapi.py

api:              ## Run the API on 127.0.0.1:8010
	cd backend && ../$(PY) -m uvicorn onconova.api.app:app --host 127.0.0.1 --port 8010 --reload

web:              ## Run the web app on 127.0.0.1:5174 (proxies /api to :8010)
	npm --prefix frontend run dev

test: test-backend test-frontend

test-backend:
	cd backend && ../$(PY) -m pytest

test-frontend:
	npm --prefix frontend test
	npm --prefix frontend run typecheck

clean-data:       ## Delete generated data, uploads, results and audit log (keeps code and configs)
	rm -rf data artifacts/models artifacts/experiments artifacts/evaluations
