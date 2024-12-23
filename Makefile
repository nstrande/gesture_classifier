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

init_git:
	@echo "Initializing git repository"
	{ \
    if [ -d .git ]; then \
        echo "Git repository already exists."; \
    else \
        git init; \
    fi \
	}

generate_annotations:
	@echo "Running annotator"
	@python src/annotation_tools/generate_annotations_data.py
	
preprocess_traning_data:
	@echo "Preprocessing data"
	@python src/data/preprocess_traning_data.py

train_model:
	@echo "Training model"
	@python src/models/train.py

run_main:
	@echo "Running main"
	@python src/main.py

convert_to_edge_ai:
	@echo "Converting to Edge AI"
	@python src/models/convert_to_edge_ai.py