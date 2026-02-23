#!/usr/bin/env python3
"""
Obsidian Knowledge Tracker
Models an Obsidian vault as a directed graph to extract mathematically rigorous
metrics regarding atomicity and connectivity, logging them to an SQLite database
and generating an interactive Plotly dashboard.

Dependencies:
    pip install pandas networkx plotly
"""

import os
import re
import sqlite3
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
VAULT_PATH = os.path.expanduser("~/Runestone/Grimoire")
DB_PATH = os.path.join(SCRIPT_PATH, "data", "tracker_metrics.db")
HTML_REPORT_PATH = os.path.join(SCRIPT_PATH, "data", "tracker_dashboard.html")


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


def update_db(metrics, db_filepath):
    """
    Upserts the latest metrics into an SQLite database.
    Using INSERT OR REPLACE prevents duplicate entries for the same day.
    """
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()

    # Create the table if it doesn't exist, using Date as the primary key
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            Date TEXT PRIMARY KEY,
            Total_Notes INTEGER,
            Total_Links INTEGER,
            Average_Degree REAL,
            Orphan_Ratio REAL,
            Mean_Word_Count REAL,
            Std_Dev_Word_Count REAL
        )
    """)

    # Upsert the day's metrics
    cursor.execute(
        """
        INSERT OR REPLACE INTO metrics (
            Date, Total_Notes, Total_Links, Average_Degree, 
            Orphan_Ratio, Mean_Word_Count, Std_Dev_Word_Count
        ) VALUES (
            :Date, :Total_Notes, :Total_Links, :Average_Degree, 
            :Orphan_Ratio, :Mean_Word_Count, :Std_Dev_Word_Count
        )
    """,
        metrics,
    )

    conn.commit()
    conn.close()


def generate_dashboard(db_filepath, html_filepath):
    """Generates a high-fidelity, scientific-grade Plotly dashboard using LaTeX."""
    # Read the historical data from SQLite
    conn = sqlite3.connect(db_filepath)
    df = pd.read_sql_query("SELECT * FROM metrics ORDER BY Date", conn)
    conn.close()

    df["Date"] = pd.to_datetime(df["Date"])

    # --- Time-Series Regularization ---
    # Enforce a strict bijective mapping to a uniform daily index to eliminate gaps/clusters.
    # We apply piecewise linear interpolation to estimate the vault state (C^0 continuity) for unobserved days.
    if not df.empty:
        # Drop intra-day duplicates as a fallback safety net
        df = df.drop_duplicates(subset=["Date"], keep="last")
        df.set_index("Date", inplace=True)
        # Resample to exactly 1 day intervals and interpolate missing values linearly
        df = df.resample("1D").asfreq().interpolate(method="time")
        df.reset_index(inplace=True)

    # --- Scientific Dark Theme Palette ---
    bg_color = "#1e2127"  # Deep One Dark inspired background
    paper_color = "#1e2127"  # Seamless background
    grid_color = "rgba(255, 255, 255, 0.08)"  # Very subtle grid
    text_color = "#E2E8F0"  # Crisp academic light gray

    # High-contrast, colorblind-friendly scientific palette (Okabe-Ito inspired, adjusted for dark mode)
    c_edges = "#56B4E9"  # Light Blue
    c_nodes = "#E69F00"  # Orange
    c_mean = "#009E73"  # Green
    c_band = "rgba(0, 158, 115, 0.15)"  # Transparent Green
    c_deg = "#CC79A7"  # Purplish Pink
    c_conn = "#0072B2"  # Darker Blue
    c_orph = "#D55E00"  # Vermillion / Red

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            r"$\text{Network Volume } (|\mathcal{V}| \text{ and } |\mathcal{E}|)$",
            r"$\text{Atomicity Discipline } (\mu_w \pm \sigma_w)$",
            r"$\text{Extensive Linking } (\langle k \rangle)$",
            r"$\text{Vault Integration } (1 - O_t)$",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    # High precision line styling
    line_style = dict(width=2, shape="spline", smoothing=0.8)
    marker_style = dict(size=5, line=dict(width=0.5, color=text_color), symbol="circle")

    # 1. Network Growth (Smooth Area Chart)
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Total_Links"],
            fill="tozeroy",
            name=r"$|\mathcal{E}| \text{ (Edges)}$",
            mode="lines+markers",
            line=dict(color=c_edges, **line_style),
            marker=marker_style,
            fillcolor=f"rgba(86, 180, 233, 0.1)",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Edges:</b> %{y:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Total_Notes"],
            fill="tozeroy",
            name=r"$|\mathcal{V}| \text{ (Vertices)}$",
            mode="lines+markers",
            line=dict(color=c_nodes, **line_style),
            marker=marker_style,
            fillcolor=f"rgba(230, 159, 0, 0.1)",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Vertices:</b> %{y:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. Atomicity Discipline (Spline with Error Bands)
    upper_bound = df["Mean_Word_Count"] + df["Std_Dev_Word_Count"]
    lower_bound = df["Mean_Word_Count"] - df["Std_Dev_Word_Count"]
    lower_bound = lower_bound.apply(lambda x: max(0, x))

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=upper_bound,
            mode="lines",
            line=dict(width=0, shape="spline", smoothing=0.8),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=lower_bound,
            mode="lines",
            line=dict(width=0, shape="spline", smoothing=0.8),
            fill="tonexty",
            fillcolor=c_band,
            name=r"$\pm 1\sigma_w \text{ Variance}$",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>-1 Std Dev:</b> %{y:.2f} words<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Mean_Word_Count"],
            mode="lines+markers",
            name=r"$\mu_w \text{ (Mean Words)}$",
            line=dict(color=c_mean, **line_style),
            marker=marker_style,
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Mean Words:</b> %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 3. Extensive Linking (Average Degree)
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Average_Degree"],
            mode="lines+markers",
            name=r"$\langle k \rangle \text{ (Avg Degree)}$",
            line=dict(color=c_deg, **line_style),
            marker=marker_style,
            fill="tozeroy",
            fillcolor="rgba(204, 121, 167, 0.1)",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Avg Degree:</b> %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 4. Integration (Modern Stacked Bar)
    connected_ratio = 1.0 - df["Orphan_Ratio"]
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=connected_ratio,
            name=r"$1 - O_t \text{ (Connected)}$",
            marker_color=c_conn,
            marker_line=dict(width=0),
            opacity=0.85,
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Connected:</b> %{y:.1%}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Orphan_Ratio"],
            name=r"$O_t \text{ (Orphans)}$",
            marker_color=c_orph,
            marker_line=dict(width=0),
            opacity=0.85,
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Orphans:</b> %{y:.1%}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # --- Global Layout & Typography ---
    fig.update_layout(
        title=dict(
            text=r"$\textbf{Grimoire Graph Topology & Atomicity Analysis}$<br><span style='font-size:14px; color:#A0AEC0'>Time-series quantification of network integrity and structural moments.</span>",
            font=dict(family="Georgia, serif", size=22, color=text_color),
            x=0.02,
            y=0.96,
        ),
        font=dict(family="Georgia, serif", color=text_color, size=13),
        paper_bgcolor=bg_color,
        plot_bgcolor=paper_color,
        barmode="stack",
        autosize=True,  # Allows the plot to elastically scale to viewport
        margin=dict(l=60, r=40, t=130, b=60),
        hovermode="closest",  # Changed from "x unified" to avoid rendering raw names
        hoverlabel=dict(
            bgcolor="#282c34",  # Elevated background for the tooltip (One Dark theme style)
            font_size=13,
            font_color="#FFFFFF",  # Explicit crisp white to fix faded text
            font_family="Georgia, serif",
            bordercolor=grid_color,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # Subplot axes styling - Maximizing Data-Ink Ratio
    axes_styling = dict(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        griddash="dot",
        zeroline=True,
        zerolinecolor="rgba(255, 255, 255, 0.2)",
        zerolinewidth=1,
        showline=True,
        linecolor="rgba(255, 255, 255, 0.3)",
        linewidth=1,
        tickfont=dict(color="#A0AEC0"),
    )

    fig.update_xaxes(**axes_styling)
    fig.update_yaxes(**axes_styling)

    # Specific y-axis formats
    fig.update_yaxes(title_text=r"$\text{Cardinality}$", row=1, col=1)
    fig.update_yaxes(title_text=r"$\text{Word Count}$", row=1, col=2)
    fig.update_yaxes(title_text=r"$\text{Degree}$", row=2, col=1)
    fig.update_yaxes(
        title_text=r"$\text{Proportion}$", tickformat=".0%", range=[0, 1], row=2, col=2
    )

    # Clean up subplot titles to ensure MathJax renders perfectly
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=16, color=text_color, family="Georgia, serif")

    # include_mathjax='cdn' is required for offline HTML to render LaTeX equations
    fig.write_html(
        html_filepath,
        include_mathjax="cdn",
        default_height="100%",
        default_width="100%",
    )
    print(f"Dashboard generated at {html_filepath}")
    webbrowser.open("file://" + html_filepath)


def main():
    print("Parsing Obsidian vault...")
    G, word_counts = parse_vault(VAULT_PATH)

    print("Calculating topological metrics...")
    metrics = calculate_metrics(G, word_counts)

    if metrics:
        # Ensure the data directory exists before writing
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        update_db(metrics, DB_PATH)
        print("Metrics successfully upserted into SQLite database.")
        generate_dashboard(DB_PATH, HTML_REPORT_PATH)
    else:
        print("Vault is empty or could not be parsed.")


if __name__ == "__main__":
    main()
