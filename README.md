# gen

A specialized language model for content generation and analysis.

## Features

- Code generation with specialized models
- Content analysis and optimization
- Performance tracking and metrics
- Real-time response generation
- Attention pattern visualization

## Setup

1. Clone the repository:
```bash
git clone https://gitlab.com/nexgencreators/gen.git
cd gen
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
gen/
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
- GitLab for version control

## Syncing Changes

To sync changes with GitLab, use the PowerShell script:
```powershell
.\sync.ps1
```

## License

MIT License
