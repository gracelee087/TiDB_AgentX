# TiDB AgentX Hackathon 2025 - Financial Question Processing System

## 🏆 Multi-Step Agentic AI Solution

A sophisticated financial question processing system that demonstrates real-world agentic AI workflows using TiDB Serverless, vector search, and OpenAI integration.

## 🚀 Features

### Multi-Step Agentic Workflow
1. **Data Ingestion**: Automatically loads and vectorizes financial documents (Guide.docx) into TiDB Serverless
2. **Intelligent Search**: Uses TiDB Vector Search with cosine similarity to find relevant document chunks
3. **AI Processing**: OpenAI GPT-4 analyzes search results and generates contextual responses
4. **Data Persistence**: All interactions, search results, and AI responses are logged in TiDB
5. **Session Management**: Maintains conversation history and context across interactions

### Technical Highlights
- **TiDB Serverless Integration**: Full vector search capabilities with cosine similarity
- **Document Processing**: Automatic chunking and vectorization of financial documents
- **Real-time Search**: Sub-second vector similarity search across document corpus
- **Comprehensive Logging**: Complete audit trail of all system interactions
- **Error Handling**: Robust error handling and user feedback

## 🛠️ Technology Stack

- **Database**: TiDB Serverless (Vector Search + Traditional SQL)
- **AI/ML**: OpenAI GPT-4, OpenAI Embeddings (text-embedding-3-large)
- **Framework**: LangChain for document processing and AI orchestration
- **Language**: Python 3.11+
- **Document Processing**: docx2txt for financial document parsing

## 📋 Prerequisites

1. **TiDB Cloud Account**: Sign up at [TiDB Cloud](https://tidbcloud.com)
2. **OpenAI API Key**: Get your API key from [OpenAI Platform](https://platform.openai.com)
3. **Python 3.11+**: Ensure Python is installed on your system

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/gracelee087/TiDB_AgentX.git
cd TiDB_AgentX
```

### 2. Install Dependencies

#### macOS
```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (PowerShell)
```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (Git-Bash)
```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
# TiDB Cloud Configuration
TIDB_HOST=your-tidb-host.tidbcloud.com
TIDB_PORT=4000
TIDB_USER=your-username
TIDB_PASSWORD=your-password
TIDB_DATABASE=your-database-name
CA_PATH=ca-cert.pem

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
```

### 4. Download TiDB CA Certificate
Download the CA certificate from your TiDB Cloud console and save as `ca-cert.pem` in the project root.

### 5. Prepare Financial Document
The project includes a sample financial document `Guide.docx` in the repository. You can use this file directly or replace it with your own financial document.

## 🚀 Running the Application

### Start the System
```bash
python main.py
```

The system will automatically:
1. Connect to TiDB Cloud
2. Create necessary tables
3. Vectorize the Guide.docx document
4. Start the interactive question processing system

### Using the System
1. Enter financial questions when prompted
2. The system will search relevant documents using vector similarity
3. AI will generate contextual responses based on found information
4. All interactions are logged in TiDB for analysis

### Example Questions
- "What are the key liquidity ratios and how are they calculated?"
- "What are the core principles of financial analysis at Unity Financial Group?"
- "How do we calculate return on equity?"

## 📊 Data Flow Architecture

```
User Question → TiDB Storage → Vector Search → Document Retrieval → 
OpenAI Processing → AI Response → TiDB Logging → User Output
```

### Database Schema
- **chat_sessions**: User session management
- **chat_messages**: Human and AI message storage
- **search_logs**: Vector search results and scores
- **vector_documents**: Document chunks with embeddings

## 🎯 Hackathon Compliance

This project demonstrates all required building blocks:

1. **✅ Ingest & Index Data**: Financial documents are automatically loaded and vectorized
2. **✅ Search Your Data**: TiDB Vector Search finds relevant document chunks
3. **✅ Chain LLM Calls**: OpenAI processes search results and generates responses
4. **✅ Multi-Step Flow**: Complete automated workflow from input to final action

## 📈 Performance Metrics

- **Vector Search Speed**: Sub-second similarity search
- **Document Processing**: Automatic chunking and vectorization
- **AI Response Time**: 2-5 seconds for complex financial queries
- **Accuracy**: High relevance scores (0.8+ similarity threshold)

## 🔧 Troubleshooting

### Common Issues
1. **No vector documents found**: Ensure Guide.docx is in the project root
2. **TiDB connection failed**: Verify your .env configuration
3. **OpenAI API errors**: Check your API key and billing status

### Debug Mode
The system provides detailed logging for troubleshooting:
- Database connection status
- Document processing progress
- Vector search results and scores
- AI response generation

## 📝 Project Structure

```
tidb-agentx-hackathon/
├── main.py              # Main application entry point
├── llm.py               # TiDB integration and AI processing
├── requirements.txt     # Python dependencies
├── Guide.docx          # Financial document (your data)
├── ca-cert.pem         # TiDB CA certificate
├── .env                # Environment configuration
└── README.md           # This file
```

## 🏅 Hackathon Submission

**TiDB Cloud Account Email**: [Your email]
**Repository URL**: https://github.com/gracelee087/TiDB_AgentX.git
**Demo Video**: [Link to your demonstration video]

## 📞 Support

For technical issues or questions about this implementation, please refer to the TiDB Cloud documentation or create an issue in this repository.

---

**Built for TiDB AgentX Hackathon 2025 - Forge Agentic AI for Real-World Impact**