import os
import sys
import time
import subprocess
from typing import Optional, Dict, Any

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class ScriptRunner:
    """Script executor, supports executing Python and Bash scripts"""

    def __init__(self, log_path: str = "data/local_logs/train.log"):
        """
        Initialize script executor
        Args:
            log_path: Base path for log files
        """
        self.base_log_path = log_path

    def _prepare_log_file(self, script_type: str) -> str:
        """
        Prepare log file
        Args:
            script_type: Script type, used for log directory naming
        Returns:
            str: Complete path to the log file
        """
        # Create log directory
        log_dir = os.path.join(self.base_log_dir, script_type)
        os.makedirs(log_dir, exist_ok=True)

        # Generate log filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(log_dir, f"{script_type}_{timestamp}.log")

    def _check_execution_env(self) -> Dict[str, str]:
        """
        Get current execution environment information, supporting docker or regular system environment
        Returns:
            Dict[str, str]: Dictionary containing environment type and detailed information
        """
        env_info = {
            "type": "system",
            "details": "Unknown environment"
        }
        
        # Check if in docker environment - first check environment variable
        if os.environ.get("IN_DOCKER_ENV") == "1":
            env_info["type"] = "docker"
            env_info["details"] = "docker-env-variable"
            return env_info
        
        # Regular system environment
        try:
            import platform
            system_info = platform.platform()
            env_info["details"] = system_info
        except Exception:
            pass
            
        return env_info

    def _check_python_version(self) -> str:
        """
        Get Python version information
        Returns:
            str: Python version information
        """
        return sys.version

    def execute_script(
        self,
        script_path: str,
        script_type: str,
        is_python: bool = False,
        args: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Execute script
        Args:
            script_path: Complete path to the script
            script_type: Script type, used for log directory naming
            is_python: Whether it is a Python script
            args: List of additional script parameters
        Returns:
            Dict[str, Any]: Execution result, including process ID, environment information and log file path
        """
        try:
            # Check if script exists
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Script does not exist: {script_path}")

            # Get execution environment information
            env_info = self._check_execution_env()
            logger.info(f"Running in environment: {env_info['type']} ({env_info['details']})")
            
            # Prepare log file
            log_file = self.base_log_path
            logger.info(f"Starting {script_type} task, log file: {log_file}")

            # Ensure script has execution permission
            os.chmod(script_path, 0o755)

            # Build command
            if is_python:
                command = [sys.executable, script_path]
            else:
                command = ["bash", script_path]

            # Add additional parameters
            if args:
                command.extend(args)

            # Record Python version (if it's a Python script)
            if is_python:
                logger.info(f"Python version: {self._check_python_version()}")

            # Execute script
            from subprocess import PIPE
            
            # Open log file
            with open(log_file, "a", buffering=1) as f:
                process = subprocess.Popen(
                    command,
                    shell=False,  # Use list form of command, no need for shell=True
                    cwd=os.getcwd(),
                    env=os.environ.copy(),
                    stdout=PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True  # This ensures text mode output
                )

                # Get process ID
                pid = process.pid
                logger.info(f"Process started, PID: {pid}")

                # Read output in real-time
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # Write to file
                        f.write(output)
                        # Write to console
                        print(output, end='', flush=True)

                # Wait for process to end
                exit_code = process.wait()
                end_message = f"Process (PID: {pid}) has ended, exit code: {exit_code}"
                logger.info(end_message)
                print(end_message)
                f.write(f"\n{end_message}\n")

            return {
                "pid": pid,
                "environment": env_info,
                "log_file": log_file,
                "exit_code": exit_code,
            }

        except Exception as e:
            error_msg = f"Failed to execute script: {str(e)}"
            logger.error(error_msg)
            raise
