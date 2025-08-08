CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    genre TEXT NOT NULL,
    confidence REAL,
    source TEXT, -- 'upload', 'live'
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);