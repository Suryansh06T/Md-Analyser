#!/usr/bin/env python3
"""
Configuration module for Obsidian Knowledge Tracker.
Contains all configuration constants and paths.
"""

import os

# --- CONFIGURATION ---
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
VAULT_PATH = os.path.expanduser("~/Runestone/Grimoire")
DB_PATH = os.path.join(SCRIPT_PATH, "tracker_metrics.db")
HTML_REPORT_PATH = os.path.join(SCRIPT_PATH, "output", "tracker_dashboard.html")

