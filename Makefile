# ── Makefile — replicación JRFM-4427748 ──────────────────────────────
# Uso: make all   (o por etapas: make data | models | eval | tables | figures)

# GNU Make 3.81 (macOS's default) does not strip trailing whitespace before
# an inline "#" comment on a ":=" assignment -- STAMPS previously carried
# trailing spaces into its value, so "$(STAMPS)/tables" etc. parsed as TWO
# space-separated prerequisite words (".stamps" and "/tables"), which Make
# then tried to build as a literal target with no rule ("No rule to make
# target `/tables'"). Comments moved to their own line to avoid the
# trailing-whitespace trap entirely.
PYTHON := .venv/bin/python
SRC    := src
# .stamps: marcadores de etapa completada
STAMPS := .stamps

.PHONY: all data models eval tables figures test clean help
.DEFAULT_GOAL := help

# Each stage's REAL target is its stamp file ($(STAMPS)/<stage>) -- this is
# what downstream stages actually depend on, so Make can skip a stage whose
# stamp is newer than its prerequisite's. `data`/`models`/`eval`/`tables`/
# `figures` are thin .PHONY aliases so `make tables` etc. still reads
# naturally. Previously every stage depended on "$(STAMPS)/<prev-stage>" as
# a prerequisite, but no rule's TARGET was ever named that (the rules were
# named "data", "models", ... instead) -- so Make had no rule to build any
# of those prerequisites and `make all` failed immediately, before running
# a single command. Apparently never run end-to-end before this fix.

$(STAMPS):
	mkdir -p $(STAMPS)

## all: pipeline completo (datos → modelos → evaluación → tablas → figuras)
all: $(STAMPS)/figures
	@echo ">> Pipeline completo. Revisa outputs/ y logs/."

## data: descarga y cachea CSV crudos, aplica split y construye el proxy ε²ₜ
data: $(STAMPS)/data
$(STAMPS)/data: | $(STAMPS)
	$(PYTHON) -m $(SRC).data.build_dataset --config config/config.yaml
	@touch $@

## models: estima/entrena los tres paneles (econométricos, ML/DL, propuesto)
models: $(STAMPS)/models
$(STAMPS)/models: $(STAMPS)/data
	$(PYTHON) -m $(SRC).models.run_econometric --config config/config.yaml
	$(PYTHON) -m $(SRC).models.garch_init      --config config/config.yaml
	Rscript R/msgarch.R config/config.yaml || (echo "ERROR: MSGARCH (R) falló — ver logs/msgarch_error.log" | tee -a logs/msgarch_error.log; exit 1)
	$(PYTHON) -m $(SRC).tuning.tune_and_train  --config config/config.yaml
	@touch $@

## eval: métricas OOS, DM+Holm, MCS, bootstrap, VaR/ES, sensibilidad de λ, degeneración, correspondencia de compuertas, escalera de ablación
eval: $(STAMPS)/eval
$(STAMPS)/eval: $(STAMPS)/models
	$(PYTHON) -m $(SRC).eval.run_all_metrics       --config config/config.yaml
	$(PYTHON) -m $(SRC).eval.degeneracy            --config config/config.yaml
	$(PYTHON) -m $(SRC).eval.gate_correspondence   --config config/config.yaml
	$(PYTHON) -m $(SRC).models.ablation_ladder     --config config/config.yaml
	@touch $@

## tables: emite Tablas 3–13, B1 y A1–A4 en .csv, .tex y .docx
tables: $(STAMPS)/tables
$(STAMPS)/tables: $(STAMPS)/eval
	$(PYTHON) -m $(SRC).reporting.build_tables --config config/config.yaml
	@echo ">> Tablas en outputs/tables/ (csv, tex, docx)."
	@touch $@

## figures: sensibilidad de λ, curvas train/val, dinámicas de compuertas
figures: $(STAMPS)/figures
$(STAMPS)/figures: $(STAMPS)/tables
	$(PYTHON) -m $(SRC).reporting.build_figures --config config/config.yaml
	@touch $@

## test: pruebas mínimas de cada módulo
test:
	$(PYTHON) -m pytest -q tests/

## clean: borra salidas y marcadores (conserva data/raw/ cacheado)
clean:
	rm -rf $(STAMPS) outputs/tables/* outputs/figures/* outputs/models/* logs/*.log logs/*.json

## help: lista los objetivos disponibles
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
