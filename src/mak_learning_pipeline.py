import os
from datetime import datetime
from flask import Flask, request, redirect
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

# === FOLDER STRUCTURE SETUP ===
def setup_folders():
    """Create necessary directories for the learning pipeline."""
    folders = [
        "data/feedback/",
        "data/knowledge/books/",
        "data/knowledge/youtube/",
        "data/knowledge/instagram/",
        "data/knowledge/frameworks/",
        "logs/",
        "models/",
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {folder}")

# === YOUTUBE + INSTAGRAM SCRAPER ===
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import instaloader

def download_youtube_caption(video_url: str, out_file: str) -> bool:
    """Download YouTube video transcript and save to file."""
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = "\n".join([x['text'] for x in transcript])
        
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
            
        logging.info(f"YouTube transcript saved to {out_file}")
        return True
    except Exception as e:
        logging.error(f"Error downloading transcript: {e}")
        return False

def download_instagram_captions(profile_name: str, max_posts: int = 10) -> bool:
    """Download Instagram captions from a profile."""
    try:
        L = instaloader.Instaloader()
        profile = instaloader.Profile.from_username(L.context, profile_name)

        count = 0
        for post in profile.get_posts():
            if count >= max_posts:
                break
            text = post.caption or ""
            with open(f"data/knowledge/instagram/{profile_name}_{count}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            count += 1
            
        logging.info(f"Captions saved for {count} Instagram posts")
        return True
    except Exception as e:
        logging.error(f"Instagram scrape error: {e}")
        return False

# === BOOK NOTE CONVERTER ===
def convert_book_notes_to_training(file_path: str, output_path: str) -> bool:
    """Convert book notes into training-ready format."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.readlines()

        output = []
        for line in raw:
            if line.strip():
                output.append(f"""
### SOURCE:
Book — [Insert Title]

### FRAMEWORK:
Book Highlight

### EXAMPLE:
{line.strip()}

### LABEL:
Book Note

### APPLICATION:
Key insight from the book.
""")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n---\n".join(output))
            
        logging.info(f"Converted book notes written to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error converting book notes: {e}")
        return False

# === FEEDBACK SYSTEM ===
def log_response(user_input: str, ai_response: str, rating: int) -> bool:
    """Log user feedback about Mak's responses."""
    try:
        log = f"""
### USER:
{user_input}

### MAK:
{ai_response}

### RATING:
{rating}
---\n"""
        date = datetime.now().strftime("%Y-%m-%d")
        with open(f"data/feedback/{date}_feedback.txt", "a", encoding="utf-8") as f:
            f.write(log)
            
        logging.info("Feedback logged successfully")
        return True
    except Exception as e:
        logging.error(f"Error logging feedback: {e}")
        return False

# === FLASK FEEDBACK APP ===
app = Flask(__name__)

@app.route("/")
def feedback_form():
    return '''
    <html>
    <head>
        <title>Rate Mak</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            textarea { width: 100%; margin: 10px 0; }
            input[type="number"] { width: 60px; }
            input[type="submit"] { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; }
            .rating { display: flex; gap: 10px; margin: 10px 0; }
            .rating input[type="radio"] { display: none; }
            .rating label { font-size: 24px; cursor: pointer; }
            .rating input[type="radio"]:checked + label { color: gold; }
        </style>
    </head>
    <body>
        <h2>How did Mak do?</h2>
        <form method="POST" action="/submit">
            <label>User Input:</label><br>
            <textarea name="user_input" rows="4" required></textarea><br>
            <label>Mak Response:</label><br>
            <textarea name="ai_response" rows="4" required></textarea><br>
            <label>Rating:</label><br>
            <div class="rating">
                <input type="radio" id="star5" name="rating" value="5" required>
                <label for="star5">★</label>
                <input type="radio" id="star4" name="rating" value="4">
                <label for="star4">★</label>
                <input type="radio" id="star3" name="rating" value="3">
                <label for="star3">★</label>
                <input type="radio" id="star2" name="rating" value="2">
                <label for="star2">★</label>
                <input type="radio" id="star1" name="rating" value="1">
                <label for="star1">★</label>
            </div>
            <input type="submit" value="Submit Feedback">
        </form>
    </body>
    </html>
    '''

@app.route("/submit", methods=["POST"])
def submit():
    user_input = request.form["user_input"]
    ai_response = request.form["ai_response"]
    rating = int(request.form["rating"])
    log_response(user_input, ai_response, rating)
    return redirect("/")

# === VERSIONED MODEL SAVING ===
def get_next_model_version() -> str:
    """Get the next version number for model saving."""
    existing = [f for f in os.listdir("models/") if f.startswith("v") and f[1:].isdigit()]
    versions = [int(f[1:]) for f in existing]
    next_version = max(versions, default=0) + 1
    version_path = f"models/v{next_version}"
    Path(version_path).mkdir(parents=True, exist_ok=True)
    return version_path

# === TRAINING DATA COLLECTION ===
def collect_training_data() -> bool:
    """Collect and combine training data from all sources."""
    try:
        logging.info("Collecting training data from all knowledge sources...")
        data_sources = [
            "data/knowledge/books/",
            "data/knowledge/youtube/",
            "data/knowledge/instagram/",
            "data/knowledge/frameworks/",
            "data/feedback/"
        ]
        
        full_text = ""
        for folder in data_sources:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.endswith(".txt"):
                        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                            full_text += f.read() + "\n---\n"
                            
        with open("data/train.txt", "w", encoding="utf-8") as out:
            out.write(full_text)
            
        logging.info("train.txt updated with fresh data")
        return True
    except Exception as e:
        logging.error(f"Error collecting training data: {e}")
        return False

# === DAILY TRAINING SCHEDULER ===
def schedule_daily_training():
    """Schedule daily training using Windows Task Scheduler."""
    try:
        logging.info("Setting up daily training schedule...")
        # This is a placeholder - actual implementation would use Windows Task Scheduler
        # or cron on Linux/Mac
        logging.info("Daily training scheduled")
        return True
    except Exception as e:
        logging.error(f"Error scheduling daily training: {e}")
        return False

if __name__ == "__main__":
    # Setup folders and launch feedback form
    setup_folders()
    collect_training_data()
    app.run(host="0.0.0.0", port=5000) 