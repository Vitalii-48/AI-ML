# task_1\enums.py
from enum import Enum

class LogFormat(str, Enum):
    MD = "md"
    JSON = "json"