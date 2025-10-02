# Gen AI-driven EDA

An AI-driven data analysis prototype built on LangGraph's Agentic workflows and Intugles data discovery capabilities to provide intelligent data exploration, hypothesis generation and testing.

- **EDA Workflows**: LangGraph-powered data analysis pipelines
- **Intelligent Hypothesis Generation**: AI-driven hypothesis creation and testing
- **Multi-Agent Architecture**: Specialized agents for different analysis tasks
- **Data Mapping and Aggregations Using Intugle**
## 🏗️ Architecture

The platform consists of several specialized nodes:

- **Univariate Analysis Node**: Statistical analysis of individual variables
- **Hypothesis Generation Node**: AI-powered hypothesis creation
- **EDA Analysis Node**: Comprehensive exploratory data analysis
- **Summarizer Node**: Intelligent result summarization

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

### 5. Set Up Intugle


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

Use the attached Jupyter Notebook and point to your data

```



### You can also use Individual nodes for specific functionalities directly

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


## 🛠️ Development

### Development Setup

For development work, install the development requirements:

```bash
pip install -r requirements_dev.txt
```

## 🤝 Contributing

Please reach to me. I would love to brainstorm and work together :)
1. Create a feature branch
2. Make your changes
3. Run tests: `pytest`
4. Submit a pull request
