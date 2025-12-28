import pkgutil
import importlib
from pathlib import Path


def auto_discover_processors():
    """Automatically discover and import all processors"""
    # get processors directory path
    processors_path = Path(__file__).parent.parent / "processors"

    # iterate over all subdirectories in processors directory
    for _, name, _ in pkgutil.iter_modules([str(processors_path)]):
        # if it is a directory and contains processor.py
        processor_file = processors_path / name / "processor.py"
        if processor_file.exists():
            module_path = f"lpm_kernel.file_data.processors.{name}.processor"
            try:
                importlib.import_module(module_path)
            except ImportError as e:
                print(f"Failed to load processor module {module_path}: {e}")
