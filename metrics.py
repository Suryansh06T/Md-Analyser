#!/usr/bin/env python3
"""
Metrics module for Obsidian Knowledge Tracker.
Computes graph theory and statistical metrics from the parsed vault.
"""

from datetime import datetime


def calculate_metrics(G, word_counts):
    """
    Computes rigorous graph theory and statistical metrics.
    """
    V = len(G.nodes)
    E = len(G.edges)

    if V == 0:
        return None

    # Graph Connectivity
    avg_degree = (2 * E) / V if V > 0 else 0

    # Integration: degree = in_degree + out_degree. If 0, it's isolated.
    isolated_nodes = [n for n, d in G.degree() if d == 0]
    orphan_ratio = len(isolated_nodes) / V

    # Atomicity: First moment (mean) and square root of the second central moment (std dev)
    counts = list(word_counts.values())
    if counts:
        mean_words = sum(counts) / len(counts)
        variance = sum((x - mean_words) ** 2 for x in counts) / len(counts)
        std_dev_words = variance**0.5
    else:
        mean_words, std_dev_words = 0, 0

    return {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Total_Notes": V,
        "Total_Links": E,
        "Average_Degree": round(avg_degree, 4),
        "Orphan_Ratio": round(orphan_ratio, 4),
        "Mean_Word_Count": round(mean_words, 2),
        "Std_Dev_Word_Count": round(std_dev_words, 2),
    }

