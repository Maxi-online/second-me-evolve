# Second Me Evolve

This repository is a **fork** of **Second Me** by **Mindverse**.

Key additions in this fork include automatic reinforcement learning (A/B + LLM-based judge), ClickHouse-backed RL logging, and an upgraded Hugging Face base model for DPO.

## What it is

**Second Me** is a “second self” system: local memory + chat interface + data generation / training pipeline.

This fork adds (high level):

- **Fully automatic self-improvement (no likes/dislikes):** A/B generation → LLM-based judge → preference pairs → DPO fine-tuning.
- **User mood / satisfaction inference** by the judge and reward logging.
- **ClickHouse required** as the only RL database for `chat_experiences`, `chat_preferences`, `chat_feedback`.
- **Automatic reinforce pipeline** on a timer/threshold: ClickHouse → dataset export → `dpo_train.py`.
- **HF base model:** `Qwen/Qwen2.5-3B-Instruct` (downloaded/used as `base_model_path`).

## Quick start (Windows / Docker Desktop)

1. Start ClickHouse and the services:

```bash
docker compose up -d clickhouse
docker compose up -d --build backend frontend
```

1. Open the UI:

- `http://localhost:3000`

## Automatic training (how it works)

- Each request to `/api/kernel2/chat` produces two candidate answers (A/B).
- The judge selects the better one → only the winner is shown to the user.
- The `(chosen, rejected)` pair is stored in ClickHouse.
- On a timer (`AUTO_REINFORCE_INTERVAL_SEC`) the system runs DPO training from accumulated pairs.

## Secrets (API keys)

This fork reads secrets from:

- `data/secrets.env` (not committed; `data/` is in `.gitignore`)

It is loaded by `docker-compose.yml` / `docker-compose-gpu.yml` via `env_file`.

## Licensing & attribution

- **Upstream (original Second Me)** is licensed under **Apache License 2.0** (see `LICENSE`).
- This repository includes upstream code plus additional changes. Licensing follows the original upstream terms for upstream code.
- Attribution for this fork is provided in `NOTICE`.
- `LICENSE-MIT` applies to **my new files / contributions** only and does not change the license of upstream code.

See: `LICENSE`, `NOTICE`, `LICENSE-MIT`.
