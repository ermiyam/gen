# Marketing LLM Project

A specialized language model for marketing content generation and analysis.

## Features

- Code generation with specialized models
- Marketing content analysis
- Performance tracking and metrics
- Real-time response generation
- Attention pattern visualization

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ermiyam/marketing-llm-project.git
cd marketing-llm-project
```

2. Create and activate virtual environment:
```bash
python -m venv tensorflow_env
source tensorflow_env/bin/activate  # On Windows: tensorflow_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the chat interface:
```bash
python src/ai_model/response_handler.py
```

## Project Structure

```
marketing-llm-project/
├── src/
│   ├── ai_model/
│   │   ├── response_handler.py
│   │   └── code_gen_model.py
│   └── api/
│       └── server.py
├── logs/
├── cache/
├── checkpoints/
├── requirements.txt
└── README.md
```

## API Endpoints

- POST `/api/chat`: Send messages and get responses
- GET `/api/stats`: Get performance metrics
- GET `/health`: Check server status

## Development

This project uses:
- Python 3.8+
- PyTorch
- Transformers
- Flask
- Cursor IDE

## License

MIT License
