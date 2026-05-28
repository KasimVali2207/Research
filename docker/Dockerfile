FROM python:3.11-slim

WORKDIR /workspace

# System deps for faiss, psycopg2, etc.
RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace
ENV HYDRA_FULL_ERROR=1

CMD ["python", "src/run_pipeline.py"]
