# 🎥 Semantic Video Retrieval System 

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://semantic-video-retrieval-system.streamlit.app/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/jayanthk82/ats)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready, multimodal video retrieval system** that matches videos to **natural language queries** using powerful vision-language models (BLIP) and Vector Databases (ChromaDB).

This version includes a clean UI, speed optimizations, and robust error handling.

---

## ✨ Key Features

- **Natural Language Search**: "Find the clip where a dog is running on grass."
- **Optimized Performance**: Adjustable frame sampling (doesn't process every single frame, making it 5-10x faster).
- **Video Summary Generation**: Automatically captions video content.
- **Clean UI**: Tabbed interface for Search, Upload, and Library management.
- **Caching**: Models load once, preventing slow reloads.

---

## 🧠 Tech Stack

| Component | Technology |
|----------|-------------|
| **Language** | Python |
| **Vision Model** | BLIP (Salesforce) |
| **Vector DB** | ChromaDB |
| **Audio Processing** | ffmpeg, pydub |
| **Frame Extraction** | OpenCV |
| **UI / Deployment** | Streamlit |

---

## 🛠️ Installation

1. **Clone the repository** (or download the files).

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

   #start the demo

   streamlit run run.py
   
