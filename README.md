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
git clone https://github.com/ermiyam/gen.git
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
- GitHub for version control

## Syncing Changes

To sync changes with GitHub, use the PowerShell script:
```powershell
.\sync.ps1
```

## License

MIT License

# Mak AI Learning System

This system automatically scrapes content from YouTube, Instagram, and TikTok, and uses it to train the Mak AI model.

## Setup

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys (when ready to use Reddit and Twitter):
```
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```

## Usage

### Starting the Learning Process

Run the following command:
```bash
python src/process_manager.py start
```

This will:
- Start the content scraper (runs every 6 hours)
- Start the model training (runs every 12 hours)
- Run both processes immediately on startup
- Save the process ID to `logs/learning_pid.txt`

### Checking Status

To check the status of the learning process:
```bash
python src/process_manager.py status
```

This will show:
- If the process is running
- Process start time
- CPU and memory usage
- Last 5 log entries

### Stopping the Learning Process

To stop the learning process:
```bash
python src/process_manager.py stop
```

## Logs

- `logs/continuous_learn.log`: Main learning process logs
- `logs/scraper.log`: Content scraper logs
- `logs/process_manager.log`: Process management logs
- `logs/learning_pid.txt`: Process ID file (automatically managed)

## Data

- `data/train.txt`: Training data collected by the scraper
- `models/`: Directory containing trained model versions

## Notes

- The scraper currently focuses on YouTube, Instagram, and TikTok
- Reddit and Twitter integration is temporarily disabled
- The system uses quality filtering to ensure only high-quality content is used for training
- Training data is automatically formatted for the model
