import os
import sys
import subprocess

req_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

if os.path.exists(req_path):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", req_path]
    )
