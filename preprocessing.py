import cv2
import math
import logging
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from sentence_transformers import SentenceTransformer

logger = logging.getLogger('preprocessing')

class VideoProcessor:
    def __init__(self):
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models on {self.device}...")

        # Load BLIP (Visual Question Answering)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)
        
        # Load Sentence Transformer (Text Embeddings)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        logger.info("Models loaded successfully.")

    def process_video(self, video_path, frame_interval=2, progress_callback=None):
        """
        Generates a summary and embedding for a video.
        
        Args:
            video_path: Path to the video file.
            frame_interval: Process 1 frame every X seconds (Higher = Faster).
            progress_callback: Streamlit progress bar callback function.
        """
        summary_text = self._generate_visual_summary(video_path, frame_interval, progress_callback)
        vector_embedding = self._generate_embedding(summary_text)
        return summary_text, vector_embedding

    def _generate_visual_summary(self, video_path, frame_interval, progress_callback):
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate stride (how many frames to skip)
        stride = int(fps * frame_interval)
        if stride < 1: stride = 1
        
        story = []
        current_frame = 0
        
        logger.info(f"Processing {video_path} (Duration: {duration:.2f}s, Interval: {frame_interval}s)")

        while True:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = vidcap.read()
            
            if not ret:
                break

            # Update progress UI if callback provided
            if progress_callback:
                progress = min(current_frame / total_frames, 1.0)
                progress_callback(progress)

            # Convert BGR (OpenCV) to RGB (PIL)
            # Resize small for speed (BLIP standard is often 384x384)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Generate Caption
            caption = self._caption_image(pil_image)
            story.append(caption)
            
            # Skip frames
            current_frame += stride

        vidcap.release()
        
        # Combine unique captions to avoid extreme repetition
        full_story = ". ".join(story)
        return full_story

    def _caption_image(self, raw_image):
        question = "describe the image in detail"
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.vqa_model.generate(**inputs)
            
        return self.processor.decode(out[0], skip_special_tokens=True)

    def _generate_embedding(self, text):
        return self.embedder.encode(text, convert_to_numpy=True)