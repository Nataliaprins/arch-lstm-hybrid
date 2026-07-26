# ── Makefile — replicación JRFM-4427748 ──────────────────────────────
# Uso: make all   (o por etapas: make data | models | eval | tables | figures)

PYTHON := .venv/bin/python
SRC    := src
# Marcadores de etapa completada.
STAMPS := .stamps

.PHONY: all data models eval tables figures test clean help
.DEFAULT_GOAL := help

$(STAMPS):
	mkdir -p $(STAMPS)

## all: pipeline completo (datos → modelos → evaluación → tablas → figuras)
all: figures
	@echo ">> Pipeline completo. Revisa outputs/ y logs/."

## data: descarga y cachea CSV crudos, aplica split y construye el proxy ε²ₜ
data: | $(STAMPS)
	$(PYTHON) -m $(SRC).data.build_dataset --config config/config.yaml
	@touch $(STAMPS)/data

## models: estima/entrena los tres paneles (econométricos, ML/DL, propuesto)
models: data
	$(PYTHON) -m $(SRC).models.run_econometric --config config/config.yaml
	@if command -v Rscript >/dev/null 2>&1; then \
		Rscript R/msgarch.R config/config.yaml || echo "WARN: MSGARCH (R) falló; continúo con el resto del pipeline."; \
	else \
		echo "WARN: Rscript no está disponible; se omite MSGARCH y continúo."; \
	fi
	$(PYTHON) -m $(SRC).tuning.tune_and_train  --config config/config.yaml
	@touch $(STAMPS)/models

## eval: métricas OOS, DM+Holm, MCS, bootstrap, VaR/ES, sensibilidad de λ
eval: models
	$(PYTHON) -m $(SRC).eval.run_all_metrics   --config config/config.yaml
	@touch $(STAMPS)/eval

## tables: emite Tablas 3–9 y 4+A1–A4 en .csv, .tex y .docx
tables: eval
	$(PYTHON) -m $(SRC).reporting.build_tables --config config/config.yaml
	@echo ">> Tablas en outputs/tables/ (csv, tex, docx)."
	@touch $(STAMPS)/tables

## figures: sensibilidad de λ, curvas train/val, dinámicas de compuertas
figures: tables
	$(PYTHON) -m $(SRC).reporting.build_figures --config config/config.yaml
	@touch $(STAMPS)/figures

## test: pruebas mínimas de cada módulo
test:
	$(PYTHON) -m pytest -q tests/

## clean: borra salidas y marcadores (conserva data/raw/ cacheado)
clean:
	rm -rf $(STAMPS) outputs/tables/* outputs/figures/* outputs/models/* logs/*.log logs/*.json

## help: lista los objetivos disponibles
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
