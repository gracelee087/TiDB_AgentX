#!/usr/bin/env python3
"""
TiDB AgentX Hackathon 2025 - Main Application
Financial Question Processing System

Workflow:
1. User enters financial question
2. Save question to TiDB
3. Search financial documents using TiDB Vector Search (cosine distance)
4. Save search results and scores to TiDB
5. Generate AI response using OpenAI based on found documents
6. Save AI response to TiDB
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any

from llm import (
    tidb_manager, 
    get_ai_response_with_logging, 
    save_ai_response,
    enhanced_retriever_search,
    add_guide_document
)
# Slack integration removed for hackathon submission

def process_question(user_question: str, session_id: str = None) -> Dict[str, Any]:
    """
    Main function to process user questions
    
    Args:
        user_question: User question
        session_id: Session ID (auto-generated if not provided)
    
    Returns:
        Processing result dictionary
    """
    if not session_id:
        session_id = f"session_{int(time.time())}"
    
    print(f"🚀 Question processing started: {session_id}")
    print(f"📝 Question: {user_question}")
    print("=" * 60)
    
    try:
        # 0. Auto-vectorize if no vector documents exist
        from sqlalchemy import text
        conn = tidb_manager.engine.connect()
        result = conn.execute(text("SELECT COUNT(*) FROM vector_documents"))
        doc_count = result.scalar()
        conn.close()
        
        if doc_count == 0:
            print("🔄 No vector documents found, vectorizing Guide.docx...")
            add_guide_document()
        
        # 1. Question processing started
        
        # 2. Save user question to TiDB
        print("1️⃣ Saving user question to TiDB...")
        tidb_manager.save_chat_message(
            session_id=session_id,
            message_type="human",
            content=user_question,
            metadata={
                "source": "terminal",
                "timestamp": datetime.now().isoformat()
            }
        )
        print("✅ Question saved successfully")
        
        # 3. Search financial documents using TiDB Vector Search (cosine distance)
        print("2️⃣ Searching financial documents using TiDB Vector Search...")
        docs, search_score = enhanced_retriever_search(user_question, session_id)
        print(f"✅ Search completed: {len(docs)} documents, average score: {search_score:.4f}")
        
        # 4. Search results confirmed
        
        # 5. Generate AI response using OpenAI based on found documents
        print("3️⃣ Generating AI response using OpenAI...")
        ai_response = get_ai_response_with_logging(user_question, session_id)
        
        # 6. Collect AI response
        response_text = ""
        for chunk in ai_response:
            if hasattr(chunk, 'content'):
                response_text += chunk.content
        
        # 7. Save AI response to TiDB
        print("4️⃣ Saving AI response to TiDB...")
        save_ai_response(response_text, session_id)
        print("✅ AI response saved successfully")
        
        # 8. AI response completed
        
        result = {
            "status": "success",
            "session_id": session_id,
            "question": user_question,
            "documents_found": len(docs),
            "search_score": search_score,
            "ai_response": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
        print("=" * 60)
        print("✅ All processing completed!")
        print(f"📊 Results: {len(docs)} documents, score: {search_score:.4f}")
        print(f"🤖 AI Response: {response_text[:100]}...")
        
        return result
        
    except Exception as e:
        error_msg = f"Error occurred during question processing: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Error handling
        
        return {
            "status": "error",
            "session_id": session_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main function - Interactive question processing"""
    print("🚀 TiDB AgentX Hackathon 2025 - Financial Question Processing System")
    print("=" * 80)
    print("Enter financial questions. Type 'quit' or 'exit' to terminate.")
    print("=" * 80)
    
    while True:
        try:
            user_question = input("\n💬 Question: ").strip()
            
            if user_question.lower() in ['quit', 'exit']:
                print("👋 Program terminated.")
                break
            
            if not user_question:
                print("❌ Please enter a question..")
                continue
            
            # Process question
            result = process_question(user_question)
            
            if result['status'] == 'success':
                print(f"\n✅ Processing completed! Session ID: {result['session_id']}")
            else:
                print(f"\n❌ Processing failed: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program terminated.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    # Auto-vectorize Guide.docx file
    print("🔄 Vectorizing Guide.docx file...")
    from llm import clear_and_revectorize_guide
    clear_and_revectorize_guide()
    print("="*80)
    
    main()