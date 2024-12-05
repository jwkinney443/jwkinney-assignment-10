# Define your virtual environment and Flask app
VENV = venv
FLASK_APP = app.py
PYTHON_VERSION = python3.11

# Install dependencies
install:
	$(PYTHON_VERSION) -m venv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip
	./$(VENV)/bin/pip install -r requirements.txt

# Run the Flask application
run:
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development ./$(VENV)/bin/flask run --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies
reinstall: clean install
