install_dev: install_miniforge install_python_requirements init_git install_pre_commit

install_miniforge:
	@echo "Installing miniforge for ARM architecture (M2)"
	@brew install miniforge

install_python_requirements:
	@echo "Installing python requirements"
	@pip install -r requirements.txt
	@echo "Install project"
	@pip install -e .

install_pre_commit:
	@echo "Installing pre-commit"
	@pre-commit install

run_pre_commit:
	@echo "Running pre-commit"
	@pre-commit run --all-files

generate_annotations:
	@echo "Running annotator"
	@python src/steps/generate_annotations_data.py
	
preprocess_training_data:
	@echo "Preprocessing data"
	@python src/steps/preprocess_training_data.py

train_model:
	@python src/steps/train.py

run_main:
	@echo "Running main"
	@python src/main.py

convert_to_edge_ai:
	@echo "Converting to Edge AI"
	@python src/steps/convert_to_edge_ai.py


# Test
pytest:
	@echo "Running pytest"
	pytest tests/