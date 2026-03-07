#!/usr/bin/env python3
"""
Database module for Obsidian Knowledge Tracker.
Handles SQLite database operations for storing metrics.
"""

import sqlite3


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

