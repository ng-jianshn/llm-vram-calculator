FROM python:3.12-slim

WORKDIR /app

# System tools needed by kubelogin (MSI auth) used by the kubeconfig exec plugin
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && KUBELOGIN_VERSION=v0.1.4 \
    && curl -fsSL -o /tmp/kubelogin.zip \
       "https://github.com/Azure/kubelogin/releases/download/${KUBELOGIN_VERSION}/kubelogin-linux-amd64.zip" \
    && apt-get install -y --no-install-recommends unzip \
    && unzip /tmp/kubelogin.zip -d /tmp/kubelogin \
    && mv /tmp/kubelogin/bin/linux_amd64/kubelogin /usr/local/bin/kubelogin \
    && chmod +x /usr/local/bin/kubelogin \
    && rm -rf /tmp/kubelogin /tmp/kubelogin.zip \
    && apt-get purge -y unzip && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements-aoai.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app_aoai.py app.py
COPY benchmark_k8s.py benchmark_k8s.py
COPY benchmark_storage.py benchmark_storage.py
COPY templates/ templates/
COPY static/ static/

# Flask listens on 5000
EXPOSE 5000

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
