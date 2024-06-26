run: src/manage.py
	python src/manage.py runserver 0.0.0.0:8000

install: requirements.txt
	pip install -r requirements.txt

install-dev: requirements.txt requirements_dev.txt 
	make install
	pip install -e .

install-dev-ju: requirements.txt 
	make install-dev
	pip install -r requirements_dev_ju.txt

install-dev-ju-nvim: requirements.txt
	make install-dev-ju
	pip install -r requirements_dev_ju_nvim.txt

regenerate_requirements: src
	pigar generate src
