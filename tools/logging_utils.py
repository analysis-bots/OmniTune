import json
import os
import pandas as pd
from typing import Any


def format_data_for_log(item: Any) -> str:
    """
    Format different data types for readable log output.
    
    Args:
        item: Any data item to be formatted for logging
        
    Returns:
        A formatted string representation of the item
    """
    if isinstance(item, pd.DataFrame):
        return item.to_string()
    elif isinstance(item, (list, dict)):
        try:
            return json.dumps(item, indent=2)
        except:
            return str(item)
    else:
        return str(item)


def ensure_log_directory(log_dir: str, task_name: str) -> str:
    """
    Ensure the task-specific log directory exists.
    
    Args:
        log_dir: Base log directory
        task_name: Name of the task
        
    Returns:
        Path to the task-specific log directory
    """
    task_log_dir = os.path.join(log_dir, task_name)
    os.makedirs(task_log_dir, exist_ok=True)
    return task_log_dir


def log_to_file(log_file_path: str, content: str, mode: str = "a"):
    """
    Log content to a specified file.
    
    Args:
        log_file_path: Path to the log file
        content: Content to be logged
        mode: File open mode ('a' for append, 'w' for write)
    """
    with open(log_file_path, mode, encoding="utf-8") as f:
        f.write(content + "\n") 