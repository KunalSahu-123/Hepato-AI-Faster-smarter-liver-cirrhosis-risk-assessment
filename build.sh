#!/usr/bin/env bash
# Render build script — runs on every deploy
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Ensure the uploads directory exists
mkdir -p static/uploads
