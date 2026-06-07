# Use a patched Python slim image to keep the runtime smaller and reduce the
# surface area for known container vulnerabilities.
FROM python:3.10.16-slim-bookworm

# Keep Python output unbuffered so container logs show up immediately.
ENV PYTHONUNBUFFERED=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System dependencies required by PDF partitioning / OCR tooling.
# poppler-utils: PDF rendering helpers
# tesseract-ocr: OCR for image text extraction
# libmagic1: file type detection used by some parsing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
	poppler-utils \
	tesseract-ocr \
	libmagic1 \
	&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so Docker can cache this layer when code
# changes but requirements do not.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code last so local edits do not invalidate dependency
# installation unless necessary.
COPY . .

# The service listens on 8000 by default.
EXPOSE 8000

# Launch the FastAPI app with uvicorn. The app is defined in `app/api.py`.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
