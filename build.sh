#!/usr/bin/env bash
# Render build script — runs on every deploy
set -o errexit

# Install Tesseract OCR and poppler for lab-report scanning
apt-get update -qq && apt-get install -y -qq tesseract-ocr poppler-utils

pip install --upgrade pip
pip install -r requirements.txt

# Ensure the uploads directory exists
mkdir -p static/uploads
