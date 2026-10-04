"""Initialize runner credentials without requiring a private config.py file."""
import importlib.util
import os
from pathlib import Path
import sys
import types

from dotenv import load_dotenv


def initialize_environment(root, model, provider, log_dir):
    load_dotenv(Path(root) / '.env')
    if 'config' in sys.modules or importlib.util.find_spec('config') is not None:
        return
    # Legacy engine imports require these names. Keep the fallback process-local;
    # credentials are read from the environment and no config file is written.
    config = types.ModuleType('config')
    for name in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'CLAUDE_API_KEY',
                 'MISTRAL_API_KEY', 'CF_API_KEY', 'CF_ACCOUNT_ID'):
        setattr(config, name, os.environ.get(name, ''))
    config.MODEL = model
    config.PROVIDER = provider
    config.LOG_DIR = str(log_dir)
    config.ENABLE_AGENT_ATTRIBUTE_STATS = False
    config.STATS_SAMPLE_SIZE = 3000
    sys.modules['config'] = config
