"""
Auto-reinforcement scheduler (ClickHouse-based).

This runs inside the backend process (daemon thread) when enabled via env vars.
It periodically checks collected preference pairs and triggers DPO training.

Defaults are SAFE: disabled unless AUTO_REINFORCE_DPO_ENABLED=1.
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime
from threading import Thread
from lpm_kernel.common.logging import logger
from lpm_kernel.api.common.script_executor import ScriptExecutor
from lpm_kernel.api.domains.kernel2.services.clickhouse_store import clickhouse_store


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


def _load_state(state_path: str) -> dict:
    try:
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def _save_state(state_path: str, state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save auto-reinforce state: {e}")


def start_auto_reinforce_dpo() -> None:
    """
    Start a daemon thread that periodically triggers DPO training from collected preferences.

    Enable with:
      AUTO_REINFORCE_DPO_ENABLED=1
    """
    enabled = os.getenv("AUTO_REINFORCE_DPO_ENABLED", "0") == "1"
    if not enabled:
        return

    interval_sec = _env_int("AUTO_REINFORCE_INTERVAL_SEC", 3600)
    min_pairs = _env_int("AUTO_REINFORCE_MIN_PAIRS", 50)

    base_model_path = os.getenv(
        "AUTO_REINFORCE_BASE_MODEL_PATH",
        os.path.join("resources", "model", "hf", "Qwen2.5-3B-Instruct"),
    )
    output_dir = os.getenv("AUTO_REINFORCE_OUTPUT_DIR", os.path.join("resources", "L2", "data", "dpo"))
    os.makedirs(output_dir, exist_ok=True)

    # Training hyperparams
    num_train_epochs = _env_int("AUTO_REINFORCE_NUM_EPOCHS", 2)
    learning_rate = _env_float("AUTO_REINFORCE_LR", 5e-6)
    batch_size = _env_int("AUTO_REINFORCE_BATCH_SIZE", 2)
    beta = _env_float("AUTO_REINFORCE_BETA", 0.1)
    lora_r = _env_int("AUTO_REINFORCE_LORA_R", 32)
    lora_alpha = _env_int("AUTO_REINFORCE_LORA_ALPHA", 64)
    lora_dropout = _env_float("AUTO_REINFORCE_LORA_DROPOUT", 0.1)

    state_path = os.getenv("AUTO_REINFORCE_STATE_PATH", os.path.join("data", "reinforce", "auto_reinforce_state.json"))
    lock_path = os.getenv("AUTO_REINFORCE_LOCK_PATH", os.path.join("data", "reinforce", "auto_reinforce.lock"))

    script_executor = ScriptExecutor()

    def loop():
        logger.info(
            f"[AUTO_REINFORCE] enabled=1 interval_sec={interval_sec} min_pairs={min_pairs} base_model_path={base_model_path}"
        )
        while True:
            try:
                time.sleep(max(5, interval_sec))

                # simple lock to prevent overlap
                if os.path.exists(lock_path):
                    continue

                pref_cnt = clickhouse_store.count_table("chat_preferences")
                if pref_cnt < min_pairs:
                    continue

                state = _load_state(state_path)
                last_trained_at = state.get("last_trained_at")
                if last_trained_at:
                    # cooldown handled by sleep interval, but keep for safety
                    pass

                # acquire lock
                os.makedirs(os.path.dirname(lock_path), exist_ok=True)
                with open(lock_path, "w", encoding="utf-8") as f:
                    f.write(datetime.now().isoformat())

                # export preferences
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                output_data_path = os.path.join(output_dir, f"dpo_from_feedback_{ts}.json")

                rows = clickhouse_store.fetch_preferences(limit=200000)
                exported = []
                for r in rows:
                    try:
                        prompt_messages = json.loads(r.get("prompt_messages") or "[]")
                    except Exception:
                        prompt_messages = []
                    exported.append({
                        "prompt": prompt_messages,
                        "chosen": r.get("chosen", ""),
                        "rejected": r.get("rejected", ""),
                    })
                with open(output_data_path, "w", encoding="utf-8") as f:
                    json.dump(exported, f, ensure_ascii=False, indent=2)

                log_file = os.path.join("logs", "train", "auto_reinforce_dpo.log")
                train_args = [
                    "--training_data_path", output_data_path,
                    "--base_model_path", base_model_path,
                    "--num_train_epochs", str(num_train_epochs),
                    "--learning_rate", str(learning_rate),
                    "--batch_size", str(batch_size),
                    "--beta", str(beta),
                    "--lora_r", str(lora_r),
                    "--lora_alpha", str(lora_alpha),
                    "--lora_dropout", str(lora_dropout),
                ]

                logger.info(f"[AUTO_REINFORCE] Starting DPO training from {len(exported)} pairs -> {output_data_path}")
                result = script_executor.execute(
                    script_path=os.path.join(os.getcwd(), "lpm_kernel", "L2", "dpo", "dpo_train.py"),
                    script_type="auto_reinforce_dpo",
                    args=train_args,
                    shell=False,
                    log_file=log_file,
                )

                state["last_trained_at"] = datetime.now().isoformat()
                state["last_output_data_path"] = output_data_path
                state["last_result"] = result
                _save_state(state_path, state)

            except Exception as e:
                logger.error(f"[AUTO_REINFORCE] error: {e}", exc_info=True)
            finally:
                try:
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                except Exception:
                    pass

    t = Thread(target=loop, daemon=True)
    t.start()

