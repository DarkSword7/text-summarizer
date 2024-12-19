import streamlit as st
import torch
from transformers import pipeline

def check_gpu_availability():
    """
    Check and return GPU availability information.
    
    Returns:
    tuple: (is_cuda_available, device_name)
    """
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0)
    return False, "No GPU available"

def summarize_text(text, max_length=150, min_length=50, use_gpu=True):
    """
    Generate a summary of the input text using a pre-trained summarization model.
    
    Args:
    text (str): Input text to be summarized
    max_length (int): Maximum length of the summary
    min_length (int): Minimum length of the summary
    use_gpu (bool): Whether to use GPU if available
    
    Returns:
    str: Summarized text
    """
    try:
        # Determine device
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Initialize the summarization pipeline with device
        summarizer = pipeline("summarization", 
                              model="facebook/bart-large-cnn", 
                              device=0 if device.type == "cuda" else -1)
        
        # Generate summary
        summary = summarizer(text, 
                             max_length=max_length, 
                             min_length=min_length, 
                             do_sample=False)[0]['summary_text']
        
        return summary
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    # Set page configuration
    st.set_page_config(page_title="AI Text Summarizer", page_icon="📝")
    
    # Check GPU availability
    gpu_available, gpu_name = check_gpu_availability()
    
    # Page title and description
    st.title("🤖 AI Text Summarizer")
    st.markdown("Generate concise summaries of your text using state-of-the-art AI")
    
    # GPU Information
    if gpu_available:
        st.success(f"🚀 GPU Detected: {gpu_name}")
    else:
        st.warning("⚠️ No GPU detected. Running on CPU (might be slower).")
    
    # GPU Usage Toggle
    use_gpu = st.toggle("Use GPU (if available)", value=gpu_available)
    
    # Input text area
    input_text = st.text_area("Enter your text here:", 
                               height=300, 
                               placeholder="Paste the text you want to summarize...")
    
    # Summarization parameters
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.slider("Maximum Summary Length", 
                               min_value=50, 
                               max_value=300, 
                               value=150, 
                               step=10)
    
    with col2:
        min_length = st.slider("Minimum Summary Length", 
                               min_value=20, 
                               max_value=max_length, 
                               value=50, 
                               step=10)
    
    # Summarize button
    if st.button("Summarize", type="primary"):
        if input_text.strip():
            # Show loading spinner
            with st.spinner("Generating summary..."):
                # Generate summary
                summary = summarize_text(input_text, max_length, min_length, use_gpu)
            
            # Display summary
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Hugging Face Transformers and Streamlit*")

if __name__ == "__main__":
    main()

# Installation Instructions for GPU Support:
# 1. CUDA and GPU Setup
#    - Ensure you have a CUDA-compatible NVIDIA GPU
#    - Install NVIDIA GPU drivers
#    - Install CUDA Toolkit (matching your GPU's capabilities)
#
# 2. Create a virtual environment:
#    python -m venv summarizer_env
#    source summarizer_env/bin/activate  # On Windows, use `summarizer_env\Scripts\activate`
#
# 3. Install PyTorch with CUDA support:
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#
# 4. Install other required packages:
#    pip install streamlit transformers
#
# 5. Run the application:
#    streamlit run text_summarizer.py