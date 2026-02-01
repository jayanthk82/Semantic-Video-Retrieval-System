import streamlit as st
import os
import time
import shutil
from database import VectorDB
from preprocessing import VideoProcessor

# --- Page Config ---
st.set_page_config(
    page_title="Semantic Video Retrieval",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Resources (Run once) ---
@st.cache_resource
def get_db():
    return VectorDB()

@st.cache_resource
def get_processor():
    return VideoProcessor()

# Initialize resources
db = get_db()
processor = get_processor()

# --- Utility Functions ---
def save_uploaded_file(uploaded_file):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    storage_dir = os.path.join(base_dir, "storage")
    os.makedirs(storage_dir, exist_ok=True)
    
    file_path = os.path.join(storage_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    frame_interval = st.slider(
        "Processing Speed (Frame Interval)", 
        min_value=1, max_value=10, value=2, 
        help="Higher values = Faster processing but less detail. (e.g., 2 means check one frame every 2 seconds)"
    )
    
    st.divider()
    
    st.subheader("ℹ️ Guide for Testing")
    with st.expander("How to use this app"):
        st.markdown("""
        1. **Go to 'Upload & Index'**.
        2. Drag & drop a short video (MP4).
        3. Click **Process Videos**.
        4. Wait for the AI to analyze the frames.
        5. **Go to 'Search'**.
        6. Type a query like *"a person running"* or *"red car"*.
        7. See matches!
        """)
    
    st.info(f"Database contains **{db.count()}** videos.")

# --- Main Interface ---
st.title("🎥 Semantic Video Retrieval System")
st.markdown("Search inside your videos using natural language.")

tab1, tab2, tab3 = st.tabs(["🔍 Search", "📤 Upload & Index", "📂 Library"])

# --- TAB 1: SEARCH ---
with tab1:
    st.markdown("### Find a Video")
    query = st.text_input("Describe what you are looking for...", placeholder="e.g., 'A dog playing in the park' or 'Chef cooking pasta'")
    
    if st.button("Search", type="primary"):
        if not query:
            st.warning("Please enter a query.")
        elif db.count() == 0:
            st.error("Database is empty! Please upload videos first.")
        else:
            with st.spinner("Searching vector database..."):
                # Embed query
                query_vector = processor._generate_embedding(query)
                results = db.search(query_vector, n_results=3)
            
            if results and results['ids']:
                st.success(f"Found {len(results['ids'][0])} relevant matches.")
                
                # Display Results
                for i, video_path in enumerate(results['ids'][0]):
                    score = results['distances'][0][i]
                    metadata = results['metadatas'][0][i]
                    summary = results['documents'][0][i]
                    
                    # Create a card for the result
                    with st.container():
                        st.markdown(f"#### Rank {i+1}")
                        col_vid, col_info = st.columns([1, 2])
                        
                        with col_vid:
                            if os.path.exists(video_path):
                                st.video(video_path)
                            else:
                                st.error(f"Video file missing: {video_path}")
                                
                        with col_info:
                            st.caption(f"Filename: {metadata.get('filename', 'Unknown')}")
                            st.caption(f"Relevance Distance: {score:.4f} (Lower is better)")
                            with st.expander("See AI Summary"):
                                st.write(summary)
                        st.divider()
            else:
                st.info("No matching videos found.")

# --- TAB 2: UPLOAD ---
with tab2:
    st.markdown("### Add Videos to Database")
    uploaded_files = st.file_uploader("Upload MP4 files", type=['mp4'], accept_multiple_files=True)
    
    if uploaded_files and st.button("🚀 Upload & Process"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file in enumerate(uploaded_files):
            status_text.write(f"Processing **{file.name}**... (This may take a moment)")
            
            # 1. Save File
            saved_path = save_uploaded_file(file)
            
            # 2. Process Video (Summary + Embedding)
            try:
                summary, vector = processor.process_video(
                    saved_path, 
                    frame_interval=frame_interval,
                    progress_callback=lambda x: progress_bar.progress(x)
                )
                
                # 3. Insert into DB
                success = db.insert_video(saved_path, summary, vector)
                
                if success:
                    st.toast(f"✅ Indexed: {file.name}")
                else:
                    st.error(f"Failed to index {file.name}")
                    
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
        
        status_text.write("✅ **All processing complete!**")
        progress_bar.progress(100)
        time.sleep(2)
        st.rerun()

# --- TAB 3: LIBRARY ---
with tab3:
    st.markdown("### Indexed Videos")
    files = db.get_all_files()
    
    if not files:
        st.info("Library is empty.")
    else:
        # Display as a table or grid
        for meta in files:
            with st.expander(f"📁 {meta.get('filename', 'Unknown')}"):
                st.json(meta)
                if st.button(f"Play {meta.get('filename')}", key=meta.get('path')):
                    st.video(meta.get('path'))