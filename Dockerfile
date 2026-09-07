FROM python:3.11-slim

# Install Tesseract OCR and poppler (for PDF-to-image conversion)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/uploads

EXPOSE 10000

CMD ["gunicorn", "wsgi:application", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "4", "--timeout", "120", "--preload"]
