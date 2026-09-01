FROM python:3.11-slim

# PaddleOCR/OpenCV load these at import time; the slim image doesn't carry them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Both dependency sets go into the one interpreter. The two-virtualenv split
# exists to keep a developer's UI environment light while the engine carries
# PaddleOCR/OpenCV/PyMuPDF (see DESIGN.md) - a container has no such problem,
# and one image is simpler than two envs inside it. app.py still shells out to
# the engine exactly as before; LOF2DATA_ENGINE_PYTHON (set in
# docker-compose.yml) just points that subprocess at this same Python.
#
# Copied before the source so a code edit doesn't re-run the ~1GB install.
COPY app/requirements.txt app/requirements.txt
COPY engine/requirements.txt engine/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt \
 && pip install --no-cache-dir -r engine/requirements.txt

COPY app/ app/
COPY engine/ engine/

EXPOSE 8501

# --server.address=0.0.0.0 so the port is reachable from outside the container.
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]
