"""
Conversation Database Module for Voice Chat Agent
Handles SQLite database operations for journaling requests and responses.
"""

import sqlite3
import datetime
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path


class ConversationDB:
    """SQLite database handler for conversation journaling."""
    
    def __init__(self, db_path: str = "conversations.db"):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT NOT NULL,
                transcription_confidence REAL,
                ai_response TEXT NOT NULL,
                response_tokens INTEGER,
                processing_time_ms INTEGER,
                session_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create conversation_metadata table for future extensibility
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                key TEXT NOT NULL,
                value TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp 
            ON conversations (timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session 
            ON conversations (session_id)
        """)
        
        conn.commit()
    
    def log_conversation(
        self,
        user_input: str,
        ai_response: str,
        transcription_confidence: Optional[float] = None,
        response_tokens: Optional[int] = None,
        processing_time_ms: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> int:
        """
        Log a conversation exchange to the database.
        
        Args:
            user_input: The user's input text (from speech transcription)
            ai_response: The AI's response text
            transcription_confidence: Confidence score from speech recognition
            response_tokens: Number of tokens in AI response
            processing_time_ms: Time taken to process the request
            session_id: Optional session identifier for grouping conversations
            
        Returns:
            The ID of the inserted conversation record
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations (
                user_input, ai_response, transcription_confidence,
                response_tokens, processing_time_ms, session_id
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_input, ai_response, transcription_confidence,
            response_tokens, processing_time_ms, session_id
        ))
        
        conversation_id = cursor.lastrowid
        conn.commit()
        return conversation_id
    
    def get_recent_conversations(
        self, 
        limit: int = 10,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent conversations from the database.
        
        Args:
            limit: Maximum number of conversations to retrieve
            session_id: Optional session ID to filter by
            
        Returns:
            List of conversation dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE session_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (session_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for AI context.
        
        Args:
            session_id: Session ID to retrieve history for
            limit: Maximum number of exchanges to retrieve
            
        Returns:
            List of conversation messages in chat format
        """
        conversations = self.get_recent_conversations(limit, session_id)
        
        # Convert to chat format (oldest first)
        history = []
        for conv in reversed(conversations):
            history.append({"role": "user", "content": conv["user_input"]})
            history.append({"role": "assistant", "content": conv["ai_response"]})
        
        return history
    
    def add_conversation_metadata(
        self,
        conversation_id: int,
        key: str,
        value: str
    ) -> None:
        """
        Add metadata to a conversation record.
        
        Args:
            conversation_id: ID of the conversation to add metadata to
            key: Metadata key
            value: Metadata value
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversation_metadata (conversation_id, key, value)
            VALUES (?, ?, ?)
        """, (conversation_id, key, value))
        
        conn.commit()
    
    def search_conversations(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search conversations by user input or AI response content.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching conversation dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM conversations 
            WHERE user_input LIKE ? OR ai_response LIKE ?
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about stored conversations.
        
        Returns:
            Dictionary containing conversation statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Total conversations
        cursor.execute("SELECT COUNT(*) as total FROM conversations")
        total = cursor.fetchone()["total"]
        
        # Conversations today
        cursor.execute("""
            SELECT COUNT(*) as today FROM conversations 
            WHERE date(timestamp) = date('now')
        """)
        today = cursor.fetchone()["today"]
        
        # Average response tokens
        cursor.execute("""
            SELECT AVG(response_tokens) as avg_tokens FROM conversations 
            WHERE response_tokens IS NOT NULL
        """)
        avg_tokens_row = cursor.fetchone()
        avg_tokens = avg_tokens_row["avg_tokens"] if avg_tokens_row["avg_tokens"] else 0
        
        # Average processing time
        cursor.execute("""
            SELECT AVG(processing_time_ms) as avg_time FROM conversations 
            WHERE processing_time_ms IS NOT NULL
        """)
        avg_time_row = cursor.fetchone()
        avg_time = avg_time_row["avg_time"] if avg_time_row["avg_time"] else 0
        
        return {
            "total_conversations": total,
            "conversations_today": today,
            "average_response_tokens": round(avg_tokens, 2),
            "average_processing_time_ms": round(avg_time, 2)
        }
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()