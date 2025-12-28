import logging
import os
import time
import sys
import torch  # Add torch import for CUDA detection
import traceback
from dataclasses import asdict

from flask import Blueprint, jsonify, request
from flask_pydantic import validate
import json as _json
from datetime import datetime
import uuid

from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.api.domains.kernel2.dto.chat_dto import (
    ChatRequest,
)
from lpm_kernel.api.domains.kernel2.services.chat_service import chat_service
from lpm_kernel.api.domains.kernel2.services.prompt_builder import (
    BasePromptStrategy,
    RoleBasedStrategy,
    KnowledgeEnhancedStrategy,
)
from lpm_kernel.api.domains.loads.services import LoadService
from lpm_kernel.api.services.local_llm_service import local_llm_service

from ...common.script_executor import ScriptExecutor
from ....configs.config import Config
from lpm_kernel.api.domains.kernel2.services.clickhouse_store import clickhouse_store
from lpm_kernel.api.domains.kernel2.dto.feedback_dto import (
    FeedbackRequest,
    PreferenceRequest,
    ReinforceFromFeedbackRequest,
)
from lpm_kernel.api.services.judge_llm_service import judge_llm_service
from threading import Thread

logger = logging.getLogger(__name__)

kernel2_bp = Blueprint("kernel2", __name__, url_prefix="/api/kernel2")

# Create script executor instance
script_executor = ScriptExecutor()


@kernel2_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    config = Config.from_env()
    app_name = config.app_name or "Service"  # Add default value to prevent None

    status = local_llm_service.get_server_status()
    if status.is_running and status.process_info:
        return jsonify(
            APIResponse.success(
                data={
                    "status": "running",
                    "pid": status.process_info.pid,
                    "cpu_percent": status.process_info.cpu_percent,
                    "memory_percent": status.process_info.memory_percent,
                    "uptime": time.time() - status.process_info.create_time,
                }
            )
        )
    else:
        return jsonify(APIResponse.success(data={"status": "stopped"}))


@kernel2_bp.route("/username", methods=["GET"])
def username():
    return jsonify(APIResponse.success(data={"username": LoadService.get_current_upload_name()}))

# read IN_DOCKER_ENV and output
@kernel2_bp.route("/docker/env", methods=["GET"])
def docker_env():
    return jsonify(APIResponse.success(data={"in_docker_env": os.getenv("IN_DOCKER_ENV")}))

@kernel2_bp.route("/llama/start", methods=["POST"])
def start_llama_server():
    """Start llama-server service"""
    try:
        # Get request parameters
        data = request.get_json()
        if not data or "model_name" not in data:
            return jsonify(APIResponse.error(message="Missing required parameter: model_name", code=400))

        model_name = data["model_name"]
        # Get optional use_gpu parameter with default value of True
        use_gpu = data.get("use_gpu", True)
        base_dir = os.getcwd()
        model_dir = os.path.join(base_dir, "resources/model/output/gguf", model_name)
        gguf_path = os.path.join(model_dir, "model.gguf")

        server_path = os.path.join(os.getcwd(), "llama.cpp/build/bin")
        if os.path.exists(os.path.join(os.getcwd(), "llama.cpp/build/bin/Release")):
            server_path = os.path.join(os.getcwd(), "llama.cpp/build/bin/Release")
            
        # Determine the executable name based on platform (.exe for Windows)
        if sys.platform.startswith("win"):
            server_executable = "llama-server.exe"
        else:
            server_executable = "llama-server"
        server_path = os.path.join(server_path, server_executable)

        # Check if model file exists
        if not os.path.exists(gguf_path):
            return jsonify(APIResponse.error(
                message=f"Model '{model_name}' GGUF file does not exist, please convert model first",
                code=400
            ))

        # Start the server using the LocalLLMService with GPU acceleration if requested
        success = local_llm_service.start_server(gguf_path, use_gpu=use_gpu)
        
        if not success:
            return jsonify(APIResponse.error(message="Failed to start llama-server", code=500))
            
        # Get updated service status
        status = local_llm_service.get_server_status()
        
        # Return success response with GPU info
        gpu_info = "with GPU acceleration" if use_gpu and torch.cuda.is_available() else "with CPU only"
        return jsonify(
            APIResponse.success(
                data={
                    "model_name": model_name,
                    "gguf_path": gguf_path,
                    "status": "running" if status.is_running else "starting",
                    "use_gpu": use_gpu and torch.cuda.is_available(),
                    "gpu_info": gpu_info
                },
                message=f"llama-server service started {gpu_info}"
            )
        )

    except Exception as e:
        error_msg = f"Failed to start llama-server: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))


# Flag to track if service is stopping
_stopping_server = False

@kernel2_bp.route("/llama/stop", methods=["POST"])
def stop_llama_server():
    """Stop llama-server service - Force immediate termination of the process"""
    global _stopping_server

    try:
        # If service is already stopping, return notification
        if _stopping_server:
            return jsonify(APIResponse.success(message="llama-server service is stopping"))

        _stopping_server = True  # Set stopping flag

        try:
            # use improved local_llm_service.stop_server() to stop all llama-server process
            status = local_llm_service.stop_server()

            # check if there are still processes running
            if status.is_running and status.process_info:
                pid = status.process_info.pid
                logger.warning(f"llama-server process still running: {pid}")
                return jsonify(APIResponse.success(
                    message="llama-server service could not be fully stopped. Please try again.",
                    data={"running_pid": pid}
                ))
            else:
                return jsonify(APIResponse.success(message="llama-server service has been stopped successfully"))

        except Exception as e:
            logger.error(f"Error while stopping llama-server: {str(e)}")
            return jsonify(APIResponse.error(message=f"Error stopping llama-server: {str(e)}", code=500))
        finally:
            _stopping_server = False

    except Exception as e:
        _stopping_server = False
        logger.error(f"Failed to stop llama-server: {str(e)}")
        return jsonify(APIResponse.error(message=f"Failed to stop llama-server: {str(e)}", code=500))


@kernel2_bp.route("/llama/status", methods=["GET"])
@validate()
def get_llama_server_status():
    """Get llama-server service status"""
    try:
        status = local_llm_service.get_server_status()
        return APIResponse.success(asdict(status))

    except Exception as e:
        logger.error(f"Error getting llama-server status: {str(e)}", exc_info=True)
        return APIResponse.error(f"Error getting llama-server status: {str(e)}")

@kernel2_bp.route("/chat", methods=["POST"])
@validate()
def chat(body: ChatRequest):
    """
    Chat interface - Stream response (OpenAI API compatible)

    Request parameters: Compatible with OpenAI Chat Completions API format
    - messages: List[Dict[str, str]], standard OpenAI message list with format:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I am a helpful assistant."},
            {"role": "user", "content": "What can you do for me?"}  
        ]
    - metadata: Dict[str, Any], additional parameters for request processing (optional):
        {
            "enable_l0_retrieval": true,  // whether to enable knowledge retrieval
            "enable_l1_retrieval": false, // whether to enable advanced knowledge retrieval
            "role_id": "uuid-string"      // optional role UUID for system customization
        }
    - stream: bool, whether to stream the response (default: True)
    - model: str, model identifier (optional, default uses configured model)
    - temperature: float, controls randomness (default: 0.1)
    - max_tokens: int, maximum tokens to generate (default: 2000)

    Response: Standard OpenAI Chat Completions API format
    For stream=true (Server-Sent Events):
    - id: str, response unique identifier
    - object: "chat.completion.chunk"
    - created: int, timestamp
    - model: str, model identifier
    - system_fingerprint: str, system fingerprint
    - choices: [
        {
          "index": 0,
          "delta": {"content": str},
          "finish_reason": null or "stop"
        }
      ]
    
    The last event will be: data: [DONE]
    
    For stream=false:
    - Complete response object with full message content
    """
    try:
        logger.info(f"Starting chat request: {body}")
        # 1. Check service status
        status = local_llm_service.get_server_status()
        if not status.is_running:
            # Format error response in OpenAI-compatible format
            error_msg = "LLama server is not running"
            logger.error(error_msg)
            error_response = {
                "error": {
                    "message": error_msg,
                    "type": "server_error",
                    "code": "service_unavailable"
                }
            }
            # Return as regular JSON response for non-stream or stream-compatible error
            if not body.stream:
                return APIResponse.error(message="Service temporarily unavailable", code=503), 503
            return local_llm_service.handle_stream_response(iter([error_response]))

        try:
            # -------------------------
            # AUTO reward from user mood (no manual likes)
            # -------------------------
            def _auto_record_mood_reward(messages):
                try:
                    last_user = None
                    prev_assistant = None
                    for i in range(len(messages) - 1, -1, -1):
                        if last_user is None and messages[i].get("role") == "user":
                            last_user = messages[i].get("content") or ""
                            for j in range(i - 1, -1, -1):
                                if messages[j].get("role") == "assistant":
                                    prev_assistant = messages[j].get("content") or ""
                                    break
                            break
                    if not last_user or not prev_assistant:
                        return

                    judge_res = judge_llm_service.infer_user_mood_reward(
                        user_message=last_user,
                        prev_assistant_message=prev_assistant,
                    )
                    reward = float(judge_res.get("reward", 0.0))
                    rating = 0
                    if reward >= 0.25:
                        rating = 1
                    elif reward <= -0.25:
                        rating = -1

                    exp_id = clickhouse_store.find_latest_experience_id_by_completion(prev_assistant)
                    if not exp_id:
                        return
                    clickhouse_store.insert_feedback({
                        "id": str(uuid.uuid4()),
                        "experience_id": exp_id,
                        "rating": int(rating),
                        "comment": f"auto_judge: mood={judge_res.get('mood')} satisfaction={judge_res.get('satisfaction')}",
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                except Exception as e:
                    logger.warning(f"Auto mood reward failed: {e}")

            Thread(target=_auto_record_mood_reward, args=(list(body.messages or []),), daemon=True).start()

            # -------------------------
            # AUTO A/B + Judge -> preferences (DPO data) with gpt-5-nano
            # -------------------------
            auto_ab_enabled = True
            if body.metadata and body.metadata.get("disable_auto_reinforce") is True:
                auto_ab_enabled = False

            if auto_ab_enabled:
                base_messages = list(body.messages or [])

                req_a = body.model_copy(deep=True)
                req_b = body.model_copy(deep=True)
                req_a.stream = False
                req_b.stream = False
                req_a.metadata = dict(req_a.metadata or {})
                req_b.metadata = dict(req_b.metadata or {})
                req_a.metadata["seed"] = int(req_a.metadata.get("seed_a", 101))
                req_b.metadata["seed"] = int(req_b.metadata.get("seed_b", 202))

                exp_a = str(uuid.uuid4())
                exp_b = str(uuid.uuid4())

                resp_a = chat_service.chat(
                    request=req_a,
                    stream=False,
                    json_response=False,
                    strategy_chain=[BasePromptStrategy, RoleBasedStrategy, KnowledgeEnhancedStrategy],
                    experience_id=exp_a,
                    experience_source="kernel2",
                )
                resp_b = chat_service.chat(
                    request=req_b,
                    stream=False,
                    json_response=False,
                    strategy_chain=[BasePromptStrategy, RoleBasedStrategy, KnowledgeEnhancedStrategy],
                    experience_id=exp_b,
                    experience_source="kernel2",
                )

                text_a = getattr(resp_a.choices[0].message, "content", "") if hasattr(resp_a, "choices") else ""
                text_b = getattr(resp_b.choices[0].message, "content", "") if hasattr(resp_b, "choices") else ""

                judge = judge_llm_service.choose_best_of_two(
                    prompt_messages=base_messages,
                    answer_a=text_a,
                    answer_b=text_b,
                )
                winner = (judge.get("winner") or "A").strip().upper()
                chosen_exp = exp_a if winner == "A" else exp_b
                chosen_text = text_a if winner == "A" else text_b
                rejected_text = text_b if winner == "A" else text_a

                # Save preference pair (ClickHouse ONLY)
                try:
                    clickhouse_store.insert_preference({
                        "id": str(uuid.uuid4()),
                        "source": "kernel2_auto_judge",
                        "model": body.model,
                        "prompt_messages": _json.dumps(base_messages, ensure_ascii=False),
                        "chosen": chosen_text,
                        "rejected": rejected_text,
                        "meta_data": _json.dumps({"method": "auto_ab_judge_gpt-5-nano", "judge": judge, "exp_a": exp_a, "exp_b": exp_b}, ensure_ascii=False),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                except Exception as e:
                    logger.warning(f"Failed to save auto preference: {e}")

                # Stream chosen text (already generated non-stream)
                def _stream_text(text: str):
                    chunk_size = 64
                    for i in range(0, len(text), chunk_size):
                        yield {"type": "chat_response", "content": text[i:i + chunk_size], "done": False}
                    yield {"type": "chat_response", "content": "", "done": True}

                if body.stream:
                    return local_llm_service.handle_stream_response(_stream_text(chosen_text), response_id_override=chosen_exp)
                return jsonify({
                    "id": chosen_exp,
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()),
                    "model": body.model or "models/lpm",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": chosen_text}, "finish_reason": "stop"}],
                })

            # Fallback to normal single-generation
            experience_id = str(uuid.uuid4())
            response = chat_service.chat(
                request=body,
                stream=body.stream,  # Respect the stream parameter from request
                json_response=False,
                strategy_chain=[BasePromptStrategy, RoleBasedStrategy, KnowledgeEnhancedStrategy],
                experience_id=experience_id,
                experience_source="kernel2",
            )

            if body.stream:
                return local_llm_service.handle_stream_response(response, response_id_override=experience_id)
            return jsonify(response)

        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Value error: {error_msg}")
            error_response = {
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error",
                    "code": "bad_request"
                }
            }
            if not body.stream:
                return jsonify(error_response), 400
            return local_llm_service.handle_stream_response(iter([error_response]))

    except Exception as e:
        error_msg = f"Request processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        error_response = {
            "error": {
                "message": error_msg,
                "type": "server_error",
                "code": "internal_server_error"
            }
        }
        if not getattr(body, 'stream', True):  # Default to stream if attribute missing
            return jsonify(error_response), 500
        return local_llm_service.handle_stream_response(iter([error_response]))


@kernel2_bp.route("/feedback", methods=["POST"])
@validate()
def submit_feedback(body: FeedbackRequest):
    """Disabled: feedback is automatic via judge (no manual likes)."""
    try:
        return jsonify(APIResponse.error(message="Manual feedback disabled: auto judge is used.", code=410))
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to submit feedback: {str(e)}", code=500))


@kernel2_bp.route("/preference", methods=["POST"])
@validate()
def submit_preference(body: PreferenceRequest):
    """Disabled: preferences are generated automatically via A/B + judge."""
    try:
        return jsonify(APIResponse.error(message="Manual preference disabled: auto judge is used.", code=410))
    except Exception as e:
        logger.error(f"Failed to submit preference: {e}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to submit preference: {str(e)}", code=500))


@kernel2_bp.route("/learning/stats", methods=["GET"])
def learning_stats():
    """Basic stats for collected RL feedback."""
    try:
        return jsonify(APIResponse.success(data={
            "experiences": clickhouse_store.count_table("chat_experiences"),
            "feedback": clickhouse_store.count_table("chat_feedback"),
            "preferences": clickhouse_store.count_table("chat_preferences"),
        }))
    except Exception as e:
        logger.error(f"Failed to get learning stats: {e}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to get learning stats: {str(e)}", code=500))


@kernel2_bp.route("/learning/reinforce_dpo", methods=["POST"])
@validate()
def reinforce_dpo(body: ReinforceFromFeedbackRequest):
    """
    Export preferences from DB into a DPO json file and start DPO training in background.
    This is the practical reinforcement step (DPO = stable policy-gradient style update).
    """
    try:
        # Export preferences (ClickHouse ONLY)
        rows = clickhouse_store.fetch_preferences(limit=100000)
        if len(rows) < body.min_pairs:
            return jsonify(APIResponse.error(
                message=f"Not enough preference pairs: have {len(rows)}, need {body.min_pairs}",
                code=400
            ))

        exported = []
        for r in rows:
            prompt_messages = _json.loads(r.get("prompt_messages") or "[]")
            exported.append({"prompt": prompt_messages, "chosen": r.get("chosen", ""), "rejected": r.get("rejected", "")})

        os.makedirs(os.path.dirname(body.output_data_path), exist_ok=True)
        with open(body.output_data_path, "w", encoding="utf-8") as f:
            _json.dump(exported, f, ensure_ascii=False, indent=2)

        # Kick off training via ScriptExecutor
        train_args = [
            "--training_data_path", body.output_data_path,
            "--base_model_path", body.base_model_path,
            "--num_train_epochs", str(body.num_train_epochs),
            "--learning_rate", str(body.learning_rate),
            "--batch_size", str(body.batch_size),
            "--beta", str(body.beta),
            "--lora_r", str(body.lora_r),
            "--lora_alpha", str(body.lora_alpha),
            "--lora_dropout", str(body.lora_dropout),
        ]
        log_file = os.path.join("logs", "train", "reinforce_dpo.log")
        result = script_executor.execute(
            script_path=os.path.join(os.getcwd(), "lpm_kernel", "L2", "dpo", "dpo_train.py"),
            script_type="reinforce_dpo",
            args=train_args,
            shell=False,
            log_file=log_file,
        )

        return jsonify(APIResponse.success(data={
            "exported_pairs": len(exported),
            "output_data_path": body.output_data_path,
            "train_result": result,
            "log_file": log_file,
        }, message="DPO reinforcement started"))
    except Exception as e:
        logger.error(f"Failed to reinforce via DPO: {e}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to reinforce via DPO: {str(e)}", code=500))


@kernel2_bp.route("/cuda/available", methods=["GET"])
def check_cuda_available():
    """Check if CUDA is available for model training/inference"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_info = {}
        
        if cuda_available:
            cuda_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0)
            }
        
        return jsonify(APIResponse.success(
            data={
                "cuda_available": cuda_available,
                "cuda_info": cuda_info
            },
            message="CUDA availability check completed"
        ))
    except Exception as e:
        error_msg = f"Error checking CUDA availability: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))
