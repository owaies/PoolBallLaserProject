#!/usr/bin/env python3
"""
Launcher script for PoolBot Laser Calibration System.
Run: python launch_calibration_app.py
"""

import sys
import os

# Ensure current directory is on PYTHONPATH
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from laser_calibration.app_gui import run_app

if __name__ == "__main__":
    print("Starting PoolBot AI Laser Calibration System...")
    run_app()
