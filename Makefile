# Generate a requirements.txt file from pyproject.toml if you work with Poetry
requirements: upgrade-pip
	pip-compile -o requirements.txt pyproject.toml --resolver=backtracking

upgrade-pip:
	python -m pip install --upgrade pip