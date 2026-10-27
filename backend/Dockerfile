FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ONCONOVA_DATA_DIR=/srv/onconova/data \
    ONCONOVA_ARTIFACTS_DIR=/srv/onconova/artifacts \
    ONCONOVA_CONFIGS_DIR=/srv/onconova/configs

RUN useradd --create-home --uid 10001 onconova
WORKDIR /srv/onconova/backend

COPY backend/pyproject.toml ./
COPY backend/onconova ./onconova
COPY backend/scripts ./scripts
COPY README.md /srv/onconova/README.md
COPY configs /srv/onconova/configs
# CPU-only image. Add the deep-learning extra with: --build-arg EXTRAS="[dl]"
ARG EXTRAS=""
RUN pip install --no-cache-dir ".${EXTRAS}" && mkdir -p /srv/onconova/data /srv/onconova/artifacts \
    && chown -R onconova /srv/onconova

USER onconova
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8010/api/health')" || exit 1
CMD ["uvicorn", "onconova.api.app:app", "--host", "0.0.0.0", "--port", "8010", "--proxy-headers"]
