"""
Search History Management System
Stores conversation history in SQLite with semantic search capabilities.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class HistorySession:
    """Represents a conversation session."""

    id: str
    title: str
    created_at: str
    updated_at: str
    tags: List[str]
    query_count: int = 0


@dataclass
class HistoryMessage:
    """Represents a single query/response in history."""

    id: str
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    paper_count: int = 0
    execution_time: float = 0.0
    relevance_threshold: float = 0.7


class HistoryManager:
    """Manages conversation history with SQLite storage."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize history manager."""
        if db_path is None:
            db_path = Path.home() / ".eeg_rag" / "history.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tags TEXT,  -- JSON array
                    query_count INTEGER DEFAULT 0
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    paper_count INTEGER DEFAULT 0,
                    execution_time REAL DEFAULT 0.0,
                    relevance_threshold REAL DEFAULT 0.7,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS citations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL,
                    pmid TEXT,
                    doi TEXT,
                    title TEXT,
                    relevance_score REAL,
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_citations_message ON citations(message_id)"
            )

            conn.commit()

    def create_session(
        self, title: str, tags: Optional[List[str]] = None
    ) -> HistorySession:
        """Create a new conversation session."""
        session_id = hashlib.md5(
            f"{title}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        now = datetime.now().isoformat()

        session = HistorySession(
            id=session_id, title=title, created_at=now, updated_at=now, tags=tags or []
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, title, created_at, updated_at, tags, query_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    session.id,
                    session.title,
                    session.created_at,
                    session.updated_at,
                    json.dumps(session.tags),
                    0,
                ),
            )
            conn.commit()

        return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        paper_count: int = 0,
        execution_time: float = 0.0,
        relevance_threshold: float = 0.7,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> HistoryMessage:
        """Add a message to a session."""
        message_id = hashlib.md5(
            f"{session_id}{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        now = datetime.now().isoformat()

        message = HistoryMessage(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=now,
            paper_count=paper_count,
            execution_time=execution_time,
            relevance_threshold=relevance_threshold,
        )

        with sqlite3.connect(self.db_path) as conn:
            # Insert message
            conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, timestamp, 
                                     paper_count, execution_time, relevance_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    message.id,
                    message.session_id,
                    message.role,
                    message.content,
                    message.timestamp,
                    message.paper_count,
                    message.execution_time,
                    message.relevance_threshold,
                ),
            )

            # Insert citations if provided
            if citations:
                for citation in citations:
                    conn.execute(
                        """
                        INSERT INTO citations (message_id, pmid, doi, title, relevance_score)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            message.id,
                            citation.get("pmid"),
                            citation.get("doi"),
                            citation.get("title"),
                            citation.get("relevance_score"),
                        ),
                    )

            # Update session
            conn.execute(
                """
                UPDATE sessions 
                SET updated_at = ?, query_count = query_count + 1
                WHERE id = ?
            """,
                (now, session_id),
            )

            conn.commit()

        return message

    def get_sessions(self, limit: int = 50) -> List[HistorySession]:
        """Get recent sessions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM sessions 
                ORDER BY updated_at DESC 
                LIMIT ?
            """,
                (limit,),
            )

            sessions = []
            for row in cursor:
                sessions.append(
                    HistorySession(
                        id=row["id"],
                        title=row["title"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        tags=json.loads(row["tags"]) if row["tags"] else [],
                        query_count=row["query_count"],
                    )
                )

            return sessions

    def get_session_messages(self, session_id: str) -> List[HistoryMessage]:
        """Get all messages for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """,
                (session_id,),
            )

            messages = []
            for row in cursor:
                messages.append(
                    HistoryMessage(
                        id=row["id"],
                        session_id=row["session_id"],
                        role=row["role"],
                        content=row["content"],
                        timestamp=row["timestamp"],
                        paper_count=row["paper_count"],
                        execution_time=row["execution_time"],
                        relevance_threshold=row["relevance_threshold"],
                    )
                )

            return messages

    def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            # Delete citations first (foreign key constraint)
            conn.execute(
                """
                DELETE FROM citations 
                WHERE message_id IN (
                    SELECT id FROM messages WHERE session_id = ?
                )
            """,
                (session_id,),
            )

            # Delete messages
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

            # Delete session
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

            conn.commit()

    def search_history(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search across all history (simple text search)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT m.*, s.title as session_title
                FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE m.content LIKE ? OR s.title LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT ?
            """,
                (f"%{query}%", f"%{query}%", limit),
            )

            results = []
            for row in cursor:
                results.append(
                    {
                        "message_id": row["id"],
                        "session_id": row["session_id"],
                        "session_title": row["session_title"],
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "paper_count": row["paper_count"],
                    }
                )

            return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Total sessions
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            stats["total_sessions"] = cursor.fetchone()[0]

            # Total queries
            cursor = conn.execute("SELECT COUNT(*) FROM messages WHERE role = 'user'")
            stats["total_queries"] = cursor.fetchone()[0]

            # Average execution time
            cursor = conn.execute(
                "SELECT AVG(execution_time) FROM messages WHERE role = 'assistant'"
            )
            stats["avg_execution_time"] = cursor.fetchone()[0] or 0.0

            # Total citations
            cursor = conn.execute("SELECT COUNT(*) FROM citations")
            stats["total_citations"] = cursor.fetchone()[0]

            return stats


# Backward compatibility functions for app_enhanced.py
def initialize_search_state():
    """Initialize search state (backward compatibility stub)."""
    import streamlit as st

    if "search_initialized" not in st.session_state:
        st.session_state.search_initialized = True


def render_search_history():
    """Render search history tab (backward compatibility stub)."""
    import streamlit as st

    st.markdown("### 📜 Search History")
    st.info(
        "Search history has been moved to the sidebar. Click the 📚 icon in the sidebar to view your conversation history."
    )

    # Show a preview of recent sessions
    if "history_manager" not in st.session_state:
        st.session_state.history_manager = HistoryManager()

    manager = st.session_state.history_manager
    stats = manager.get_statistics()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sessions", stats["total_sessions"])
    with col2:
        st.metric("Total Queries", stats["total_queries"])
    with col3:
        avg_time = stats["avg_execution_time"]
        st.metric("Avg Response Time", f"{avg_time:.2f}s" if avg_time else "N/A")

    st.divider()

    # Show recent sessions
    sessions = manager.get_sessions(limit=10)
    if sessions:
        st.markdown("#### Recent Sessions")
        for session in sessions:
            with st.expander(f"📄 {session.title}"):
                st.markdown(f"**Created:** {session.created_at}")
                st.markdown(f"**Queries:** {session.query_count}")
                if session.tags:
                    st.markdown(f"**Tags:** {', '.join(session.tags)}")

                # Show messages
                messages = manager.get_session_messages(session.id)
                for msg in messages:
                    role_icon = "🙋" if msg.role == "user" else "🤖"
                    st.markdown(
                        f"{role_icon} **{msg.role.title()}:** {msg.content[:200]}..."
                    )
    else:
        st.info("No search history yet. Run a query to start building your history!")
