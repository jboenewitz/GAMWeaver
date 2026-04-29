FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Needed for git-based dependencies (igann) and sklearn runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu \
    && pip install -r backend/requirements.txt

COPY backend backend
COPY bike.csv bike.csv

RUN chmod +x backend/startup.sh

EXPOSE 8000

CMD ["./backend/startup.sh"]
