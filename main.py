#!/usr/bin/env python3
"""
Obsidian Knowledge Tracker - Main Entry Point
Models an Obsidian vault as a directed graph to extract mathematically rigorous
metrics regarding atomicity and connectivity, logging them to an SQLite database
and generating an interactive Plotly dashboard.

This is a wrapper that imports functionality from modular components.

Dependencies:
    pip install pandas networkx plotly

Main module for Obsidian Knowledge Tracker.
Coordinates the parsing, metrics calculation, database update, and dashboard generation.
"""

import os
import webbrowser
from config import VAULT_PATH, DB_PATH, HTML_REPORT_PATH
from parser import parse_vault
from metrics import calculate_metrics
from database import update_db
from dashboard import generate_dashboard


def main():
    print("Parsing Obsidian vault...")
    G, word_counts = parse_vault(VAULT_PATH)

    print("Calculating topological metrics...")
    metrics = calculate_metrics(G, word_counts)

    if metrics:
        # Ensure the output directory exists before writing
        os.makedirs(os.path.dirname(HTML_REPORT_PATH), exist_ok=True)

        update_db(metrics, DB_PATH)
        print("Metrics successfully upserted into SQLite database.")
        generate_dashboard(DB_PATH, HTML_REPORT_PATH)
        webbrowser.open("file://" + HTML_REPORT_PATH)
    else:
        print("Vault is empty or could not be parsed.")


from main import main

if __name__ == "__main__":
    main()
