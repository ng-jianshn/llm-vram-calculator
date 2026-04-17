FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements-aoai.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app_aoai.py app.py
COPY templates/ templates/
COPY static/ static/

# Flask listens on 5000
EXPOSE 5000

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
