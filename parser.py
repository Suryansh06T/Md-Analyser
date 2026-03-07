#!/usr/bin/env python3
"""
Parser module for Obsidian Knowledge Tracker.
Parses Obsidian vault to build graph and compute word counts.
"""

import re
from pathlib import Path
import networkx as nx


def parse_vault(vault_dir):
    """
    Parses the vault directory to build a directed graph G(V, E)
    and compute the word count w(v) for all v in V.
    """
    G = nx.DiGraph()
    word_counts = {}

    # Regex to find [[Internal Links]].
    # The split('|')[0] handles aliased links like [[Real Note|Alias text]]
    link_pattern = re.compile(r"\[\[(.*?)\]\]")

    vault_path = Path(vault_dir)
    if not vault_path.exists():
        raise FileNotFoundError(f"Vault path {vault_dir} does not exist.")

    for filepath in vault_path.rglob("*.md"):
        # Ignore hidden directories like .obsidian
        if ".obsidian" in filepath.parts:
            continue

        node_name = filepath.stem

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Calculate w(v)
            words = len(content.split())
            word_counts[node_name] = words

            # Ensure node exists even if it has no links
            if not G.has_node(node_name):
                G.add_node(node_name)

            # Extract E_t and add to graph
            links = link_pattern.findall(content)
            for link in links:
                target_node = link.split("|")[0].strip()
                # Ignore empty links or links to headings inside the same file
                if target_node and not target_node.startswith("#"):
                    # Remove heading links from the target e.g. Note#Heading
                    target_node = target_node.split("#")[0]
                    G.add_edge(node_name, target_node)

        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}")

    return G, word_counts

