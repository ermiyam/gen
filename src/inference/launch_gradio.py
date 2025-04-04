import os
import logging
from pathlib import Path
from datetime import datetime
from rich.console import Console
from src.inference.gradio_interface import GenGradioInterface

def main():
    # Setup console
    console = Console()
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gradio_launch_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Create assets directory if it doesn't exist
        assets_dir = Path("assets")
        assets_dir.mkdir(exist_ok=True)
        
        # Check for Gen avatar
        avatar_path = assets_dir / "gen_avatar.png"
        if not avatar_path.exists():
            console.print("[yellow]Warning: Gen avatar not found. Using default avatar.[/yellow]")
        
        # Initialize and launch interface
        console.print("[bold green]🚀 Launching Gen Marketing Assistant...[/bold green]")
        
        interface = GenGradioInterface(
            model_path="models/mistral-gen",
            log_dir="data/response_logs"
        )
        
        # Launch with share option (for public access)
        interface.launch(share=True)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        logger.error(f"Error launching interface: {str(e)}")
        raise

if __name__ == "__main__":
    main() 