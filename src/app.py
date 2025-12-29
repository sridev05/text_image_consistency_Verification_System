import streamlit as st
from PIL import Image
import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
import time
import re

# ---------- 1. PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="Semantic Validator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 2. STYLED UI (CSS) ----------
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    
    /* VERDICT CARD STYLES */
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        min-height: 80px;
    }
    .v-success { background: rgba(34, 197, 94, 0.15); border: 2px solid #22c55e; color: #22c55e; }
    .v-warning { background: rgba(234, 179, 8, 0.15); border: 2px solid #eab308; color: #eab308; }
    .v-error   { background: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; color: #ef4444; }
    
    .v-title { font-size: 22px; font-weight: 800; text-transform: uppercase; margin-bottom: 5px; }
    .v-desc { font-size: 14px; color: #e2e8f0; font-weight: 400; }

    /* TAG STYLES */
    .tag-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; min-height: 30px; }
    .tag {
        background-color: #1f2937;
        color: #e2e8f0;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 12px;
        border: 1px solid #374151;
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #111827; }
    .stButton > button { width: 100%; background-color: #3b82f6; color: white; border: none; font-weight: bold; }
    .stButton > button:hover { background-color: #2563eb; }
    </style>
    """, unsafe_allow_html=True)

# ---------- 3. LOAD MODELS ----------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return clip_model, clip_processor, blip_model, blip_processor, device

clip_model, clip_processor, blip_model, blip_processor, device = load_models()

# ---------- 4. PROCESSING FUNCTION ----------
def process_image(image, user_text):
    """Process image and return all results at once"""
    start_time = time.time()
    
    # BLIP (Caption)
    inputs = blip_processor(image, text="a photo of", return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=40, num_beams=5, do_sample=False)
    ai_caption = blip_processor.decode(out[0], skip_special_tokens=True).replace("a photo of", "").strip()
    
    # CLIP (Score)
    clip_inputs = clip_processor(text=[user_text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**clip_inputs)
    img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    score = (img_emb @ txt_emb[0].T).item()
    
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    
    # Extract Keywords
    ignore_words = ['a', 'an', 'the', 'of', 'in', 'on', 'with', 'is', 'photo']
    keywords = [word for word in ai_caption.split() if word.lower() not in ignore_words and len(word) > 2]
    
    # Determine Verdict
    if score >= 0.28:
        v_class = "v-success"
        v_text = "Match Confirmed"
        v_desc = "High confidence alignment detected."
    elif score >= 0.20:
        v_class = "v-warning"
        v_text = "Partial Match"
        v_desc = "Some elements align, but ambiguity exists."
    else:
        v_class = "v-error"
        v_text = "Mismatch"
        v_desc = "The image does not match the description."
    
    return {
        'score': score,
        'ai_caption': ai_caption,
        'keywords': keywords,
        'processing_time': processing_time,
        'v_class': v_class,
        'v_text': v_text,
        'v_desc': v_desc
    }

# ---------- 5. SIDEBAR INPUTS ----------
with st.sidebar:
    st.title("👁️ Semantic Validator")
    st.markdown("---")
    
    with st.form("input_form"):
        st.subheader("1. Source Image")
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("2. Target Description")
        user_text = st.text_area("", placeholder="What should this image show?", height=100)
        
        st.markdown("---")
        submit = st.form_submit_button("🚀 Verify Consistency")

# ---------- 6. MAIN DASHBOARD ----------
if not submit:
    # Stable Empty State
    st.markdown("""
    <div style='text-align: center; margin-top: 15vh; color: #64748b;'>
        <h1>Ready to Analyze</h1>
        <p>Please use the sidebar to upload an image and description.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    if not uploaded_file or not user_text:
        st.warning("⚠️ Please provide both an image and a text description.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        
        # CREATE LAYOUT STRUCTURE FIRST (before processing)
        col_img, col_spacer, col_result = st.columns([1, 0.1, 1.2])
        
        # --- COLUMN 1: IMAGE (Display immediately) ---
        with col_img:
            st.markdown("### 🖼️ Visual Source")
            st.image(image, use_container_width=True, caption="Input Image")

        # --- COLUMN 2: PROCESS AND DISPLAY RESULTS ---
        with col_result:
            st.markdown("### ⚖️ Analysis Report")
            
            # Create a container for the loading state
            result_container = st.container()
            
            with result_container:
                # Show loading message
                loading_placeholder = st.empty()
                loading_placeholder.markdown("""
                <div style='text-align: center; padding: 40px; color: #64748b;'>
                    <div style='font-size: 18px; margin-bottom: 10px;'>🔄 Processing semantics...</div>
                    <div style='font-size: 14px;'>Analyzing image and text alignment</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Process the image
                results = process_image(image, user_text)
                
                # Clear loading and show results
                loading_placeholder.empty()
                
                # 1. THE VERDICT CARD
                st.markdown(f"""
                <div class="verdict-box {results['v_class']}">
                    <div class="v-title">{results['v_text']}</div>
                    <div class="v-desc">{results['v_desc']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 2. DETAILS SECTION
                st.markdown("#### 🧠 AI Perception")
                st.info(f"**Description:** \"{results['ai_caption']}\"")
                
                # Visual Tags
                st.markdown("**Detected Key Elements:**")
                tags_html = "".join([f"<span class='tag'>{k}</span>" for k in results['keywords']])
                st.markdown(f"<div class='tag-container'>{tags_html}</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Technical Metrics (Expander)
                with st.expander("🛠️ Advanced Metrics"):
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Raw Similarity", f"{round(results['score'], 4)}")
                    with m2:
                        st.metric("Processing Time", f"{results['processing_time']}s")
                    
                    st.caption(f"Model: CLIP-ViT-Base + BLIP-Base | Device: {device.upper()}")