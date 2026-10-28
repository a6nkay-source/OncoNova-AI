# Deployment guide

## Local development (tested)

```bash
make setup && make setup-dl
```

```bash
make api
```

```bash
make web
```

- API: `http://127.0.0.1:8010` (OpenAPI UI at `/docs`). Web: `http://127.0.0.1:5174`, which proxies `/api` to the API. Set `ONCONOVA_API_URL` to point the dev proxy elsewhere.
- Defaults: `ONCONOVA_AUTH_MODE=dev` (localhost requests only; caller treated as a local admin). Data in `./data`, models and reports in `./artifacts`.
- Apple-silicon GPUs are used automatically for U-Net training/inference (MPS), then CUDA, then CPU.

## Configuration

All settings are `ONCONOVA_*` environment variables (or `.env`; see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `ONCONOVA_AUTH_MODE` | `dev` | `dev` or `api_key` |
| `ONCONOVA_API_KEYS_FILE` | unset | YAML file of hashed keys (below) |
| `ONCONOVA_DATA_DIR` / `_ARTIFACTS_DIR` / `_CONFIGS_DIR` | repo folders | Storage locations |
| `ONCONOVA_MAX_UPLOAD_MB` | `512` | Per-upload limit, enforced while streaming |
| `ONCONOVA_MAX_ZIP_MEMBERS` / `_MAX_UNCOMPRESSED_MB` | `4000` / `2048` | ZIP limits |
| `ONCONOVA_PSEUDONYM_SALT` | generated into `data/.pseudonym_salt` | HMAC salt for DICOM pseudonyms |
| `ONCONOVA_ALLOWED_ORIGINS` | `["http://localhost:5174","http://127.0.0.1:5174"]` | CORS (JSON list) |
| `ONCONOVA_GENERATE_DEMO_CASES` | `true` | Create DEMO DATA cases at start-up |

### API keys

Store only SHA-256 hashes:

```bash
.venv/bin/python -c "import secrets,hashlib; k=secrets.token_urlsafe(32); print('key:', k); print('sha256:', hashlib.sha256(k.encode()).hexdigest())"
```

```yaml
# configs/api_keys.yaml  (git-ignored)
keys:
  - sha256: <hex>
    user: alice
    role: researcher   # viewer | researcher | admin
```

Give each person their raw key through a secure channel. Users paste it under Models & Data → Safety and access; the browser sends it as `X-API-Key`.

## Docker (written, not yet tested in this repository)

Docker was not available on the development machine, so these files have **not been built or run**. Expect to fix small issues on first build.

```bash
cp .env.example .env    # set ONCONOVA_PSEUDONYM_SALT and ONCONOVA_AUTH_MODE=api_key
```

```bash
docker compose up --build
```

- `api`: non-root user, read-only root filesystem, dropped capabilities, `no-new-privileges`, data in a named volume, artifacts and configs mounted read-only, no published port.
- `web`: nginx serving the built UI on `127.0.0.1:8080`, with a strict Content-Security-Policy and an `/api` reverse proxy.
- Add the deep-learning extra with `--build-arg EXTRAS="[dl]"` (large image).

## Hardening checklist before any shared deployment

- [ ] `api_key` auth, unique keys per user, least-privilege roles
- [ ] TLS-terminating reverse proxy and network access restricted to the research group
- [ ] Encrypted disk for `data/`; backups that follow your data-governance rules
- [ ] Retention policy: delete uploads and derived results when no longer needed (UI/API delete, or `make clean-data`)
- [ ] Review the de-identification profile with your privacy officer; check images for burned-in text
- [ ] Ship `audit.jsonl` to your log store and monitor denied requests
- [ ] Ethics/IRB approval and data-use agreements for any non-public data
- [ ] Keep the research-only banner and labels; do not connect to clinical systems (PACS write-back, TPS, record-and-verify, treatment machines)
