import warnings
# Ignore Pydantic v2 warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
# Use TiDB Vector Search instead of Pinecone

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Few-shot examples are not currently used
store = {}


class TiDBCloudManager:
    """Class for TiDB Cloud connection and data management"""
    
    def __init__(self):
        self.engine = None
        self.session = None
        self._connect()
    
    def _connect(self):
        """Connect to TiDB Cloud."""
        try:
            # Get TiDB Cloud connection info from environment variables
            tidb_host = os.getenv('TIDB_HOST')
            tidb_port = int(os.getenv('TIDB_PORT', '4000'))
            tidb_user = os.getenv('TIDB_USER')
            tidb_password = os.getenv('TIDB_PASSWORD')
            tidb_database = os.getenv('TIDB_DATABASE')
            
            if not all([tidb_host, tidb_user, tidb_password, tidb_database]):
                raise ValueError("TiDB connection parameters not found in environment variables")
            
            # Generate TiDB Cloud connection URL (using CA certificate)
            # Get CA file path from environment variables
            ca_path = os.getenv('CA_PATH', 'ca-cert.pem')
            
            if os.path.exists(ca_path):
                # Use SSL if CA certificate exists
                connection_url = f"mysql+pymysql://{tidb_user}:{tidb_password}@{tidb_host}:{tidb_port}/{tidb_database}?ssl_ca={ca_path}&ssl_verify_cert=true&ssl_verify_identity=true"
                print(f"Using CA certificate: {ca_path}")
            else:
                # Disable SSL if CA certificate not found
                connection_url = f"mysql+pymysql://{tidb_user}:{tidb_password}@{tidb_host}:{tidb_port}/{tidb_database}?ssl_disabled=false"
                print(f"CA certificate not found: {ca_path}")
            
            self.engine = create_engine(connection_url, echo=False)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            
            # Create tables
            self._create_tables()
            print("TiDB Cloud connection successful!")
            
        except Exception as e:
            print(f"TiDB Cloud connection failed: {e}")
            self.engine = None
            self.session = None
    
    def _create_tables(self):
        """Create necessary tables."""
        if not self.engine:
            return
            
        try:
            with self.engine.connect() as conn:
                # Chat sessions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        metadata JSON
                    )
                """))
                
                # Chat messages table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(255),
                        message_type ENUM('human', 'ai') NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSON,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                    )
                """))
                
                # Search logs table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS search_logs (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(255),
                        query TEXT NOT NULL,
                        retrieved_docs JSON,
                        search_score FLOAT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                    )
                """))
                
                # Vector documents table (for TiDB Vector Search)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS vector_documents (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding VECTOR(3072),
                        metadata JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Vector index creation
        # Vector index is automatically created in TiDB
        # conn.execute(text("""
        #     CREATE INDEX IF NOT EXISTS idx_vector_embedding 
        #     ON vector_documents (embedding) USING VECTOR
        # """))
                
                conn.commit()
                print("Table creation completed!")
                
        except Exception as e:
            print(f"Table creation failed: {e}")
    
    def save_chat_message(self, session_id: str, message_type: str, content: str, metadata: Optional[Dict] = None):
        """Save chat message to TiDB."""
        if not self.session:
            return False
            
        try:
            # Create session if it doesn't exist
            self._ensure_session_exists(session_id)
            
            # Save message
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO chat_messages (session_id, message_type, content, metadata)
                    VALUES (:session_id, :message_type, :content, :metadata)
                """), {
                    'session_id': session_id,
                    'message_type': message_type,
                    'content': content,
                    'metadata': json.dumps(metadata) if metadata else None
                })
                conn.commit()
            return True
            
        except Exception as e:
            print(f"Message save failed: {e}")
            return False
    
    def save_search_log(self, session_id: str, query: str, retrieved_docs: list, search_score: float):
        """Save search log to TiDB."""
        if not self.session:
            return False
            
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO search_logs (session_id, query, retrieved_docs, search_score)
                    VALUES (:session_id, :query, :retrieved_docs, :search_score)
                """), {
                    'session_id': session_id,
                    'query': query,
                    'retrieved_docs': json.dumps(retrieved_docs),
                    'search_score': search_score
                })
                conn.commit()
            return True
            
        except Exception as e:
            print(f"검색 로그 저장 실패: {e}")
            return False
    
    def _ensure_session_exists(self, session_id: str):
        """Check if session exists and create if not."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM chat_sessions WHERE session_id = :session_id
                """), {'session_id': session_id})
                
                if result.scalar() == 0:
                    conn.execute(text("""
                        INSERT INTO chat_sessions (session_id, user_id, metadata)
                        VALUES (:session_id, :user_id, :metadata)
                    """), {
                        'session_id': session_id,
                        'user_id': 'default_user',
                        'metadata': json.dumps({'created_by': 'system'})
                    })
                    conn.commit()
                    
        except Exception as e:
            print(f"Session creation failed: {e}")
    
    def get_chat_history(self, session_id: str, limit: int = 10) -> list:
        """Get chat history from TiDB."""
        if not self.session:
            return []
            
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT message_type, content, timestamp, metadata
                    FROM chat_messages 
                    WHERE session_id = :session_id 
                    ORDER BY timestamp DESC 
                    LIMIT :limit
                """), {'session_id': session_id, 'limit': limit})
                
                messages = []
                for row in result:
                    messages.append({
                        'type': row[0],
                        'content': row[1],
                        'timestamp': row[2],
                        'metadata': json.loads(row[3]) if row[3] else None
                    })
                
                return list(reversed(messages))  # Sort by time
                
        except Exception as e:
            print(f"Chat history retrieval failed: {e}")
            return []
    
    def save_vector_document(self, content: str, embedding: list, metadata: Optional[Dict] = None):
        """Save vector document to TiDB."""
        if not self.session:
            return False
            
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO vector_documents (content, embedding, metadata)
                    VALUES (:content, :embedding, :metadata)
                """), {
                    'content': content,
                    'embedding': json.dumps(embedding),  # Store vector as JSON
                    'metadata': json.dumps(metadata) if metadata else None
                })
                conn.commit()
            return True
            
        except Exception as e:
            print(f"Vector document save failed: {e}")
            return False
    
    def search_vector_documents(self, query_embedding: list, limit: int = 6, threshold: float = 0.5):
        """Search documents in TiDB."""
        if not self.session:
            return [], 0.0
            
        try:
            with self.engine.connect() as conn:
                # First check total document count
                count_result = conn.execute(text("SELECT COUNT(*) FROM vector_documents"))
                total_docs = count_result.scalar()
                print(f"🔍 Total {total_docs} documents found in database")
                
                if total_docs == 0:
                    print("❌ No vector documents found. Please vectorize Guide.docx first.")
                    print("💡 Solution: python -c \"from llm import add_guide_document; add_guide_document()\"")
                    return [], 0.0
                
                # Get all documents and calculate similarity in Python
                result = conn.execute(text("""
                    SELECT id, content, metadata, embedding
                    FROM vector_documents 
                """))
                
                documents = []
                similarities = []
                
                for row in result:
                    # Parse vector from TiDB
                    try:
                        doc_embedding = json.loads(row[3]) if row[3] else []
                        if not doc_embedding:
                            continue
                            
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(query_embedding, doc_embedding)
                        
                        # Select only documents above threshold
                        if similarity >= threshold:
                            doc = {
                                'id': row[0],
                                'content': row[1],
                                'metadata': json.loads(row[2]) if row[2] else {},
                                'similarity': similarity
                            }
                            documents.append(doc)
                            similarities.append(similarity)
                            print(f"  📄 Document {len(documents)}: {row[1][:50]}... (score: {similarity:.3f})")
                            
                    except Exception as e:
                        print(f"Vector parsing failed: {e}")
                        continue
                
                # Sort by similarity and select top limit documents
                documents.sort(key=lambda x: x['similarity'], reverse=True)
                documents = documents[:limit]
                similarities = [doc['similarity'] for doc in documents]
                
                # Calculate average similarity
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                print(f"🔍 Search results: {len(documents)} documents returned, average score: {avg_similarity:.3f}")
                return documents, avg_similarity
                
        except Exception as e:
            print(f"Vector search failed: {e}")
            return [], 0.0
    
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        if len(vec1) != len(vec2):
            return 0.0
            
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate vector magnitude
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        # Return cosine similarity (0~1 range)
        return dot_product / (magnitude1 * magnitude2)
    
    def clear_vector_documents(self, source: str = None):
        """Delete vector documents. If source is specified, delete only that source."""
        if not self.session:
            return False
            
        try:
            with self.engine.connect() as conn:
                if source:
                    # Delete only documents from specific source
                    result = conn.execute(text("""
                        DELETE FROM vector_documents 
                        WHERE JSON_EXTRACT(metadata, '$.source') = :source
                    """), {'source': source})
                    print(f"🗑️ Deleted {result.rowcount} vector documents related to {source}")
                else:
                    # Delete all vector documents
                    result = conn.execute(text("DELETE FROM vector_documents"))
                    print(f"🗑️ Deleted all {result.rowcount} vector documents")
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Vector document deletion failed: {e}")
            return False
    
    def clear_test_data(self):
        """Delete test data."""
        if not self.session:
            return False
            
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    DELETE FROM vector_documents 
                    WHERE JSON_EXTRACT(metadata, '$.source') = 'test_data'
                """))
                print(f"🗑️ Deleted {result.rowcount} test data entries")
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Test data deletion failed: {e}")
            return False
    
    def close(self):
        """Close connection."""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()


# Create TiDB Cloud manager instance
tidb_manager = TiDBCloudManager()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for the given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        
        # Load existing chat history from TiDB
        try:
            tidb_messages = tidb_manager.get_chat_history(session_id, limit=50)
            for msg in tidb_messages:
                if msg['type'] == 'human':
                    store[session_id].add_user_message(msg['content'])
                elif msg['type'] == 'ai':
                    store[session_id].add_ai_message(msg['content'])
        except Exception as e:
            print(f"Failed to load chat history from TiDB: {e}")
    
    return store[session_id]


def get_retriever():
    """
    Return a retriever that uses TiDB Vector Search.
    """
    # Use search function instead of retriever since TiDB Vector Search is directly implemented
    return None


def enhanced_retriever_search(query: str, session_id: str = None):
    """Search documents in TiDB and return results."""
    try:
        # Generate OpenAI embedding
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        query_embedding = embedding.embed_query(query)
        
        # Search documents in TiDB
        documents, avg_similarity = tidb_manager.search_vector_documents(
            query_embedding=query_embedding,
            limit=6,
            threshold=0.1
        )
        
        # Convert search results to Document objects
        from langchain_core.documents import Document
        docs = []
        
        for doc in documents:
            langchain_doc = Document(
                page_content=doc['content'],
                metadata=doc['metadata']
            )
            docs.append(langchain_doc)
        
        # Save search log to TiDB
        if session_id and tidb_manager:
            try:
                retrieved_docs = []
                for doc in documents:
                    retrieved_docs.append({
                        'content': doc['content'][:500],
                        'metadata': doc['metadata'],
                        'score': doc['similarity']
                    })
                
                tidb_manager.save_search_log(
                    session_id=session_id,
                    query=query,
                    retrieved_docs=retrieved_docs,
                    search_score=avg_similarity
                )
            except Exception as e:
                print(f"Search log save failed: {e}")
        
        return docs, avg_similarity
    except Exception as e:
        print(f"Search execution failed: {e}")
        return [], 0.0


def get_history_retriever():
    """Create a retriever that reconstructs search queries considering chat history."""
    # Use search function instead of retriever since TiDB Vector Search is directly implemented
    return None


def get_llm(model='gpt-4o'):
    """Create main LLM model object."""
    llm = ChatOpenAI(model=model)
    return llm


def get_rag_chain():
    """
    Reconstruct RAG chain and prompts.
    Directly implemented using TiDB Vector Search.
    """
    llm = get_llm()
    
    system_prompt = (
        "You are an expert financial analyst. Answer the user's questions about Financial Reporting Standards and Employee Handbook."
        "Please use the provided document to answer the question, and if you cannot find the answer, just say you don't know."
        "Always start your response with 'Regarding your financial question...' and end with 'is the answer.'"
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Simple RAG chain using TiDB Vector Search
    def rag_chain_func(inputs):
        # Perform search
        docs, score = enhanced_retriever_search(inputs["input"])
        
        # Generate context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate prompt
        prompt = qa_prompt.format(
            context=context,
            chat_history=inputs.get("chat_history", []),
            input=inputs["input"]
        )
        
        # Call LLM
        response = llm.invoke(prompt)
        return response
    
    return rag_chain_func


def get_ai_response(user_message, session_id="abc123"):
    """Return AI response to user message in streaming format."""
    # Save user message to TiDB
    if tidb_manager:
        tidb_manager.save_chat_message(
            session_id=session_id,
            message_type="human",
            content=user_message,
            metadata={"timestamp": datetime.now().isoformat()}
        )
    
    rag_chain = get_rag_chain()
    
    # Simple response generation using TiDB Vector Search
    response = rag_chain({
        "input": user_message,
        "chat_history": []
    })
    
    # Return in streaming format
    def stream_response():
        yield response
    
    return stream_response()


def get_ai_response_with_logging(user_message, session_id="abc123"):
    """AI response function with TiDB logging"""
    # Save user message to TiDB
    if tidb_manager:
        tidb_manager.save_chat_message(
            session_id=session_id,
            message_type="human",
            content=user_message,
            metadata={"timestamp": datetime.now().isoformat()}
        )
    
    # Perform TiDB Vector Search and logging
    docs, score = enhanced_retriever_search(user_message, session_id)
    
    # Execute RAG chain
    rag_chain = get_rag_chain()
    
    # Simple response generation using TiDB Vector Search
    response = rag_chain({
        "input": user_message,
        "chat_history": []
    })
    
    # Return in streaming format
    def stream_response():
        yield response
    
    return stream_response()


def save_ai_response(ai_message, session_id="abc123"):
    """Save AI response to TiDB."""
    if tidb_manager:
        tidb_manager.save_chat_message(
            session_id=session_id,
            message_type="ai",
            content=ai_message,
            metadata={"timestamp": datetime.now().isoformat()}
        )

def add_guide_document():
    """Vectorize Guide.docx file and save to TiDB."""
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Load Guide.docx file
        loader = Docx2txtLoader("Guide.docx")
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Generate embeddings
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        
        print(f"📄 Guide.docx loaded successfully: {len(chunks)} chunks")
        
        # Vectorize and save each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Vectorize chunk
                chunk_embedding = embedding.embed_query(chunk.page_content)
                
                # Save to TiDB
                success = tidb_manager.save_vector_document(
                    content=chunk.page_content,
                    embedding=chunk_embedding,
                    metadata={
                        "source": "Guide.docx",
                        "chunk_index": i,
                        "metadata": chunk.metadata
                    }
                )
                
                if success:
                    print(f"✅ Chunk {i+1}/{len(chunks)} saved successfully: {chunk.page_content[:50]}...")
                else:
                    print(f"❌ Chunk {i+1} save failed")
                    
            except Exception as e:
                print(f"❌ Chunk {i+1} processing failed: {e}")
        
        print(f"🎉 Guide.docx vectorization completed! Total {len(chunks)} chunks saved")
        
    except Exception as e:
        print(f"❌ Guide.docx load failed: {e}")
        print("💡 Please check if Guide.docx file is in the project folder.")

def clear_and_revectorize_guide():
    """Delete existing Guide.docx vector data and re-vectorize."""
    print("🔄 Guide.docx re-vectorization started...")
    
    # 1. Delete existing Guide.docx vector data
    print("🗑️ Deleting existing Guide.docx vector data...")
    success = tidb_manager.clear_vector_documents(source="Guide.docx")
    
    if not success:
        print("❌ Existing vector data deletion failed")
        return False
    
    # 2. Check if Guide.docx file exists
    if not os.path.exists("Guide.docx"):
        print("❌ Guide.docx file not found.")
        print("💡 Please place Guide.docx file in the project folder.")
        return False
    
    # 3. Re-vectorize
    print("📄 Re-vectorizing Guide.docx file...")
    add_guide_document()
    
    print("✅ Guide.docx re-vectorization completed!")
    return True

