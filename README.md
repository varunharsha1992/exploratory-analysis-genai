# Data Querying AI

A powerful AI-driven data analysis platform that combines LangGraph workflows with Intugle's semantic search capabilities to provide intelligent data exploration and hypothesis generation.

## 🚀 Features

- **Automated EDA Workflows**: LangGraph-powered data analysis pipelines
- **Intelligent Hypothesis Generation**: AI-driven hypothesis creation and testing
- **Semantic Search**: Natural language querying of your data assets
- **Multi-Agent Architecture**: Specialized agents for different analysis tasks
- **Vector Database Integration**: Qdrant-powered semantic search
- **Flexible Data Sources**: Support for CSV, Excel, and various data formats

## 🏗️ Architecture

The platform consists of several specialized agents:

- **Univariate Analysis Agent**: Statistical analysis of individual variables
- **Hypothesis Generation Agent**: AI-powered hypothesis creation
- **EDA Analysis Agent**: Comprehensive exploratory data analysis
- **Summarizer Agent**: Intelligent result summarization
- **Worker Loop Agent**: Orchestrates multi-hypothesis testing

## 📋 Prerequisites

- **Python 3.10+** (recommended)
- **Conda** (Anaconda or Miniconda)
- **Git** (for cloning repositories)
- **Docker** (for Qdrant vector database)

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Data Querying AI"
```

### 2. Create Environment

```bash
conda create -n liexp python=3.10 -y
conda activate liexp
```

### 3. Install Dependencies

For essential functionality (recommended):
```bash
pip install -r requirements_no_uvloop.txt
```

For full functionality with all optional packages:
```bash
pip install -r requirements.txt
```

For development work:
```bash
pip install -r requirements_dev.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional Configuration
QDRANT_URL=http://localhost:6333
MONGO_ATLAS_SEARCH_INDEX_ENABLED_DB=your_mongodb_connection_string_here
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/embedding-001
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
```

### 5. Set Up Intugle (Optional but Recommended)

If you have access to the Intugle repository:

```bash
# Clone Intugle repository
git clone <intugle-repo-url> data-tools

# Install Intugle from local source
cd data-tools
pip install -e .
cd ..
```

### 6. Initialize Intugle

```bash
python utils/setup_intugle.py
```

## 🧪 Testing Your Installation

### 1. Test Basic Imports

```bash
python -c "import pandas, numpy, langchain; print('✅ Basic imports successful')"
```

### 2. Test Intugle Setup

```bash
python -c "from utils.setup_intugle import is_intugle_available; print(f'Intugle available: {is_intugle_available()}')"
```

### 3. Run Sample Analysis

```bash
python -c "
from EDA.workflow.workflow_execution import execute_eda_workflow
import asyncio

async def test_workflow():
    result = await execute_eda_workflow(
        target_variable='sales',
        eda_request='Analyze sales patterns',
        domain_context='FMCG retail data'
    )
    print('✅ Workflow test successful')

asyncio.run(test_workflow())
"
```

## 📊 Usage Examples

### Basic EDA Workflow

```python
import asyncio
from EDA.workflow.workflow_execution import execute_eda_workflow

async def run_analysis():
    result = await execute_eda_workflow(
        target_variable="sales_volume",
        eda_request="Analyze factors affecting sales volume",
        domain_context="FMCG retail industry",
        hypothesis_limit=5
    )
    return result

# Run the analysis
result = asyncio.run(run_analysis())
print(result)
```

### Using Individual Agents

```python
from EDA.agents.univariate_analysis.univariate_analysis import UnivariateAnalysisAgent
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Initialize agent
agent = UnivariateAnalysisAgent(target_variable="sales")

# Run analysis
results = agent.analyze_variable(df)
print(results)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes | - |
| `GOOGLE_API_KEY` | Google API key for Gemini models | Yes | - |
| `QDRANT_URL` | Qdrant vector database URL | No | `http://localhost:6333` |
| `MONGO_ATLAS_SEARCH_INDEX_ENABLED_DB` | MongoDB connection string | No | - |
| `EMBEDDING_PROVIDER` | Embedding provider (gemini/openai) | No | `gemini` |
| `EMBEDDING_MODEL` | Embedding model name | No | `models/embedding-001` |
| `MODEL_PROVIDER` | Default model provider | No | `openai` |
| `MODEL_NAME` | Default model name | No | `gpt-4o-mini` |

### Agent Configuration

Each agent can be configured through YAML files in the `EDA/agents/{agent_name}/config.yaml` directory. Key configuration options:

- **Model Settings**: Choose between OpenAI and Google models
- **Analysis Parameters**: Customize analysis behavior
- **Timeout Settings**: Control processing timeouts
- **Output Formats**: Configure result formatting

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
**Problem**: `ModuleNotFoundError` for various packages  
**Solution**: Ensure you're in the correct conda environment
```bash
conda activate liexp
pip install -r requirements_no_uvloop.txt
```

#### Google Cloud Import Errors
**Problem**: `ImportError: cannot import name 'storage' from 'google.cloud'`  
**Solution**: Install the correct Google Cloud packages
```bash
pip install google-cloud-storage google-auth
```

#### Intugle Not Found
**Problem**: Intugle components not available  
**Solution**: Ensure Intugle is properly installed from local source
```bash
cd data-tools
pip install -e .
```

#### Environment Variables Not Loaded
**Problem**: API keys not found  
**Solution**: Ensure `.env` file exists and is in the project root
```bash
ls -la .env  # Should show the file
cat .env     # Should show your API keys
```

#### Qdrant Connection Issues
**Problem**: Cannot connect to Qdrant  
**Solution**: Start Qdrant Docker container
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 🛠️ Development

### Development Setup

For development work, install the development requirements:

```bash
pip install -r requirements_dev.txt
```

This includes:
- **Code Quality**: black, isort, flake8, pylint, mypy
- **Testing**: pytest, pytest-cov, pytest-mock
- **Documentation**: sphinx, mkdocs
- **Development Tools**: jupyter, ipython, rich

### Running Tests

```bash
# Run all tests
pytest

# Run specific agent tests
pytest EDA/agents/univariate_analysis/test_univariate_agent.py -v

# Run with coverage
pytest --cov=EDA
```

### Code Quality

```bash
# Format code
black EDA/
isort EDA/

# Lint code
flake8 EDA/
pylint EDA/
```

## 🚀 Production Deployment

For production deployment:

1. **Use minimal requirements**: `requirements_no_uvloop.txt`
2. **Environment variables**: Use secure environment variable management
3. **Vector database**: Use Qdrant Cloud or self-hosted Qdrant
4. **Monitoring**: Set up proper logging and monitoring
5. **Containerization**: Use Docker for consistent deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements_no_uvloop.txt .
RUN pip install -r requirements_no_uvloop.txt

COPY . .
CMD ["python", "EDA/workflow/workflow_execution.py"]
```

## 📚 Documentation

- **Agent Documentation**: Each agent has detailed documentation in its directory
- **Workflow Documentation**: See `EDA/workflow/` for workflow details
- **Configuration**: Agent configurations are in `EDA/agents/{agent_name}/config.yaml`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:

1. Check this README
2. Review the troubleshooting section
3. Check the project's issue tracker
4. Contact the development team

---

**Note**: This project requires access to the Intugle repository for full functionality. Contact the project maintainers for access.
