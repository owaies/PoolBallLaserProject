#!/usr/bin/env python3
"""
Launcher for PoolBot Laser Calibration System in Google Chrome.
Starts the local web server and automatically opens Google Chrome.
"""

import os
import sys
import time
import threading
import webbrowser
import subprocess

# Ensure project root is in sys.path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from laser_calibration.web_app import app

PORT = 5000
URL = f"http://127.0.0.1:{PORT}"

def open_browser():
    time.sleep(1.2)  # Wait for Flask to bind
    print(f"\nOpening {URL} in Google Chrome...")
    try:
        # Attempt to open Chrome directly on Windows
        subprocess.run(["cmd.exe", "/c", "start", "chrome", URL], check=False)
    except Exception:
        webbrowser.open(URL)

if __name__ == "__main__":
    print("=" * 65)
    print(f" Starting PoolBot Web Calibration Dashboard on {URL}")
    print("=" * 65)

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
