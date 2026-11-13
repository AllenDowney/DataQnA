PROJECT_NAME = DataQnA
PYTHON_INTERPRETER = python

create_environment:
	mamba env create -f environment.yml
	@echo ">>> env created. Activate with:\nconda activate $(PROJECT_NAME)"

update_environment:
	mamba env update -f environment.yml --prune

delete_environment:
	mamba env remove --name $(PROJECT_NAME)

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 code
	black --check --config pyproject.toml code

format:
	black --config pyproject.toml code

tests:
	pytest --nbmake nb/*.ipynb
