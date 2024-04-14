.PHONY: create_environment requirements dev_requirements clean data build_documentation serve_documentation train

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = dog_breed_identification
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Install Developer Python Dependencies
dev_requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install .["dev"]

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Process raw data into processed data
data:
	python $(PROJECT_NAME)/data/make_dataset.py

## Train model
train:
	python $(PROJECT_NAME)/train_model.py $(if $(EPOCHS),--num_epochs $(EPOCHS)) $(if $(BATCHSIZE),--batch_size $(BATCHSIZE)) $(if $(LR),--learning_rate $(LR)) $(if $(MODEL_NAME),--model_name $(MODEL_NAME))

#################################################################################
# Documentation RULES                                                           #
#################################################################################

## Build documentation
build_documentation: dev_requirements
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation: dev_requirements
	mkdocs serve --config-file docs/mkdocs.yaml

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help
# Self-documenting Makefile script
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
    	h; \
    	s/.*//; \
    	:doc" \
        -e "H; \
        n; \
        s/^## //; \
        t doc" \
        -e "s/:.*//; \
        G; \
        s/\\n## /---/; \
        s/\\n/ /g; \
        p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
    	-v ncol=$$(tput cols) \
        -v indent=19 \
        -v col_on="$$(tput setaf 6)" \
        -v col_off="$$(tput sgr0)" \
	'{ \
    	printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
    	n = split($$2, words, " "); \
    	line_length = ncol - indent; \
    	for (i = 1; i <= n; i++) { \
            line_length -= length(words[i]) + 1; \
            if (line_length <= 0) { \
                line_length = ncol - indent - length(words[i]) - 1; \
                printf "\n%*s ", -indent, " "; \
            } \
            printf "%s ", words[i]; \
        } \
        printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')