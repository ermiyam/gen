import gradio as gr
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from rich.console import Console
from rich.markdown import Markdown
import subprocess
import os
import speech_recognition as sr
from googletrans import Translator
from langdetect import detect

from src.inference.chat_with_gen import GenChat
from src.inference.response_logger import ResponseLogger, ResponseRating
from src.inference.rag_system import RAGSystem, DocumentType

class GenGradioInterface:
    """Gradio interface for Gen with response logging and tools."""
    
    def __init__(
        self,
        model_path: str = "models/mistral-gen",
        log_dir: str = "data/response_logs"
    ):
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.gen_chat = GenChat(model_path=model_path)
        self.response_logger = ResponseLogger(log_dir=log_dir)
        self.rag_system = RAGSystem()
        self.translator = Translator()
        
        # Create Gradio interface
        self.interface = self._create_interface()
    
    def _create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="Gen Marketing Assistant", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🚀 Gen Marketing Assistant
            Your AI-powered marketing companion. Create, analyze, and automate your marketing content!
            """)
            
            # Main dashboard with tabs
            with gr.Tabs():
                # Chat Tab
                with gr.Tab("💬 Talk to Gen"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Chat interface
                            chatbot = gr.Chatbot(
                                [],
                                elem_id="chatbot",
                                bubble_full_width=False,
                                avatar_images=(None, "assets/gen_avatar.png"),
                                height=600
                            )
                            
                            with gr.Row():
                                txt = gr.Textbox(
                                    show_label=False,
                                    placeholder="Ask Gen anything about marketing...",
                                    container=False
                                )
                                submit_btn = gr.Button("Send", variant="primary")
                            
                            # Voice input
                            with gr.Row():
                                audio_input = gr.Audio(
                                    source="microphone",
                                    type="filepath",
                                    label="Voice Input"
                                )
                                voice_btn = gr.Button("🎤 Use Voice")
                            
                            # Rating interface
                            with gr.Row():
                                rating = gr.Radio(
                                    choices=["✅ Good", "❌ Bad"],
                                    label="Rate this response",
                                    interactive=True
                                )
                                stars = gr.Slider(
                                    minimum=1,
                                    maximum=5,
                                    value=3,
                                    step=1,
                                    label="Stars",
                                    interactive=True
                                )
                            
                            feedback = gr.Textbox(
                                label="Feedback (optional)",
                                placeholder="What did you like/dislike about this response?",
                                lines=2
                            )
                            
                            # Stats display
                            stats = gr.JSON(
                                label="Response Statistics",
                                value=self.response_logger.get_stats()
                            )
                        
                        with gr.Column(scale=1):
                            # Marketing tools
                            gr.Markdown("### 🛠️ Marketing Tools")
                            
                            with gr.Tab("Content Creation"):
                                platform = gr.Dropdown(
                                    choices=["Instagram", "TikTok", "LinkedIn", "Twitter"],
                                    label="Platform",
                                    value="Instagram"
                                )
                                content_type = gr.Dropdown(
                                    choices=["Caption", "Hashtags", "Strategy", "Script"],
                                    label="Content Type",
                                    value="Caption"
                                )
                                generate_btn = gr.Button("Generate", variant="secondary")
                            
                            with gr.Tab("Settings"):
                                temperature = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.7,
                                    step=0.1,
                                    label="Temperature (Creativity)"
                                )
                                max_length = gr.Slider(
                                    minimum=100,
                                    maximum=1000,
                                    value=512,
                                    step=50,
                                    label="Max Response Length"
                                )
                                language = gr.Dropdown(
                                    choices=["Auto", "English", "Spanish", "French", "German"],
                                    label="Response Language",
                                    value="Auto"
                                )
                
                # Video Editing Tab
                with gr.Tab("🎬 Edit Video"):
                    with gr.Row():
                        with gr.Column():
                            video_input = gr.Video(label="Upload Video")
                            script_input = gr.Textbox(
                                label="Video Script",
                                placeholder="Enter or generate a script for your video...",
                                lines=5
                            )
                            with gr.Row():
                                generate_script_btn = gr.Button("Generate Script")
                                edit_video_btn = gr.Button("Edit Video", variant="primary")
                        
                        with gr.Column():
                            video_output = gr.Video(label="Edited Video")
                            status = gr.Textbox(label="Status")
                
                # Automations Tab
                with gr.Tab("⚡ Automations"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📅 Content Calendar")
                            schedule_date = gr.Date(label="Schedule Date")
                            schedule_time = gr.Time(label="Schedule Time")
                            platform_select = gr.Dropdown(
                                choices=["Instagram", "TikTok", "LinkedIn", "Twitter"],
                                label="Platform"
                            )
                            content_type = gr.Dropdown(
                                choices=["Post", "Story", "Reel", "Carousel"],
                                label="Content Type"
                            )
                            schedule_btn = gr.Button("Schedule Post", variant="primary")
                        
                        with gr.Column():
                            gr.Markdown("### 🔄 Automated Tasks")
                            auto_hashtags = gr.Checkbox(label="Auto-generate hashtags")
                            auto_caption = gr.Checkbox(label="Auto-generate caption")
                            auto_analytics = gr.Checkbox(label="Auto-analyze performance")
                            save_settings_btn = gr.Button("Save Automation Settings")
                
                # Analytics Tab
                with gr.Tab("📊 Analytics"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📈 Performance Metrics")
                            platform_analytics = gr.Dropdown(
                                choices=["Instagram", "TikTok", "LinkedIn", "Twitter"],
                                label="Select Platform"
                            )
                            metric_type = gr.Dropdown(
                                choices=["Engagement", "Reach", "Conversions", "ROI"],
                                label="Metric Type"
                            )
                            date_range = gr.Dropdown(
                                choices=["Last 7 days", "Last 30 days", "Last 90 days", "Custom"],
                                label="Date Range"
                            )
                            generate_report_btn = gr.Button("Generate Report", variant="primary")
                        
                        with gr.Column():
                            metrics_display = gr.JSON(label="Metrics")
                            chart_output = gr.Plot(label="Performance Chart")
            
            # Event handlers
            def user(user_message, history):
                return "", history + [[user_message, None]]
            
            def bot(history, temperature, max_length, language):
                # Update generation config
                self.gen_chat.temperature = temperature
                self.gen_chat.max_length = max_length
                
                # Get user message
                user_message = history[-1][0]
                
                # Detect language if auto
                if language == "Auto":
                    try:
                        detected_lang = detect(user_message)
                        target_lang = "en"  # Default to English
                        if detected_lang != "en":
                            # Translate to English for model
                            user_message = self.translator.translate(
                                user_message,
                                dest="en"
                            ).text
                    except:
                        target_lang = "en"
                else:
                    target_lang = language.lower()[:2]
                
                # Get relevant context from RAG
                context = self.rag_system.get_relevant_context(user_message)
                
                # Generate response
                prompt = self.gen_chat._format_prompt(user_message, context)
                response = self.gen_chat._generate_response(prompt)
                
                # Translate response if needed
                if target_lang != "en":
                    response = self.translator.translate(
                        response,
                        dest=target_lang
                    ).text
                
                # Log response
                self.response_logger.log_response(
                    prompt=user_message,
                    response=response,
                    metadata={
                        "temperature": temperature,
                        "max_length": max_length,
                        "language": target_lang,
                        "context_used": bool(context)
                    }
                )
                
                # Add to RAG if good response
                if self.response_logger.get_last_rating() == ResponseRating.GOOD:
                    self.rag_system.add_document(
                        doc_type=DocumentType.SCRIPT,
                        content=response,
                        metadata={
                            "prompt": user_message,
                            "rating": "good"
                        }
                    )
                
                history[-1][1] = response
                return history
            
            def process_voice(audio):
                try:
                    # Initialize recognizer
                    recognizer = sr.Recognizer()
                    
                    # Load audio file
                    with sr.AudioFile(audio) as source:
                        audio_data = recognizer.record(source)
                        
                        # Recognize speech
                        text = recognizer.recognize_google(audio_data)
                        
                        return text
                        
                except Exception as e:
                    self.logger.error(f"Error processing voice input: {str(e)}")
                    return ""
            
            def rate_response(history, rating, stars, feedback):
                if not history or not history[-1][1]:
                    return history, gr.update(value=None), gr.update(value=None)
                
                # Log rating
                self.response_logger.log_response(
                    prompt=history[-1][0],
                    response=history[-1][1],
                    rating=ResponseRating.GOOD if rating == "✅ Good" else ResponseRating.BAD,
                    stars=stars,
                    feedback=feedback
                )
                
                # Update stats
                return history, gr.update(value=None), gr.update(value=self.response_logger.get_stats())
            
            def generate_video_script(video):
                # Placeholder for video script generation
                return "Generated script for your video..."
            
            def edit_video(video, script):
                # Placeholder for video editing
                return video, "Video editing completed!"
            
            def schedule_post(date, time, platform, content_type):
                # Placeholder for post scheduling
                return f"Post scheduled for {date} at {time} on {platform}"
            
            def generate_report(platform, metric_type, date_range):
                # Placeholder for analytics report
                return {
                    "engagement_rate": 0.05,
                    "total_reach": 10000,
                    "conversions": 100
                }, None  # Placeholder for chart
            
            # Connect components
            submit_btn.click(
                user,
                [txt, chatbot],
                [txt, chatbot]
            ).then(
                bot,
                [chatbot, temperature, max_length, language],
                chatbot
            )
            
            voice_btn.click(
                process_voice,
                [audio_input],
                [txt]
            )
            
            rating.change(
                rate_response,
                [chatbot, rating, stars, feedback],
                [chatbot, feedback, stats]
            )
            
            generate_script_btn.click(
                generate_video_script,
                [video_input],
                [script_input]
            )
            
            edit_video_btn.click(
                edit_video,
                [video_input, script_input],
                [video_output, status]
            )
            
            schedule_btn.click(
                schedule_post,
                [schedule_date, schedule_time, platform_select, content_type],
                [status]
            )
            
            generate_report_btn.click(
                generate_report,
                [platform_analytics, metric_type, date_range],
                [metrics_display, chart_output]
            )
        
        return interface
    
    def launch(self, share: bool = False):
        """Launch the Gradio interface."""
        self.interface.launch(share=share) 