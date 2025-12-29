import streamlit as st
from PIL import Image
import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
import time

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
    /* MAIN BACKGROUND & FONT */
    .stApp {
        background-color: #0e1117;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* CUSTOM 'VERDICT' CARDS */
    .verdict-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .verdict-success {
        background: linear-gradient(135deg, #14532d 0%, #166534 100%); /* Deep Green */
        border: 2px solid #22c55e;
    }
    .verdict-warning {
        background: linear-gradient(135deg, #713f12 0%, #854d0e 100%); /* Deep Orange */
        border: 2px solid #eab308;
    }
    .verdict-error {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); /* Deep Red */
        border: 2px solid #ef4444;
    }
    
    .verdict-title {
        color: white;
        font-size: 24px;
        font-weight: 800;
        text-transform: uppercase;
        margin: 0;
        letter-spacing: 1px;
    }
    .verdict-desc {
        color: #e2e8f0;
        font-size: 14px;
        margin-top: 5px;
    }

    /* CUSTOM BAR CONTAINER */
    .bar-container {
        background-color: #334155;
        border-radius: 50px;
        height: 12px;
        width: 100%;
        margin-top: 15px;
        position: relative;
    }
    .bar-fill {
        height: 100%;
        border-radius: 50px;
        transition: width 0.5s ease;
    }
    .bar-labels {
        display: flex;
        justify-content: space-between;
        color: #94a3b8;
        font-size: 10px;
        margin-top: 5px;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }
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

# ---------- 4. SIDEBAR INPUTS ----------
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

# ---------- 5. MAIN DASHBOARD ----------
if not submit:
    # Empty State
    st.markdown("""
    <div style='text-align: center; margin-top: 100px; color: #64748b;'>
        <h1>Ready to Verify</h1>
        <p>Upload an image and description in the sidebar to start.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    if not uploaded_file or not user_text:
        st.warning("⚠️ Please provide both an image and a text description.")
    else:
        # Layout Columns
        col_img, col_result = st.columns([1, 1.2], gap="large")
        
        image = Image.open(uploaded_file).convert("RGB")
        
        # --- COLUMN 1: IMAGE ---
        with col_img:
            st.markdown("### 🖼️ Visual Source")
            st.image(image, use_container_width=True, caption="Input Image")

        # --- COLUMN 2: ANALYSIS ---
        with col_result:
            with st.spinner("Analyzing semantics..."):
                time.sleep(0.5) # UX Pause
                
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

            st.markdown("### ⚖️ Alignment Verdict")
            
            # --- LOGIC FOR VERDICT STAMP (No Percentages) ---
            if score >= 0.28:
                html_card = f"""
                <div class="verdict-box verdict-success">
                    <div class="verdict-title">✅ Match Confirmed</div>
                    <div class="verdict-desc">High confidence alignment detected.</div>
                </div>
                """
                bar_color = "#22c55e" # Green
            elif score >= 0.20:
                html_card = f"""
                <div class="verdict-box verdict-warning">
                    <div class="verdict-title">⚠️ Partial Match</div>
                    <div class="verdict-desc">Some elements align, but ambiguity exists.</div>
                </div>
                """
                bar_color = "#eab308" # Yellow/Orange
            else:
                html_card = f"""
                <div class="verdict-box verdict-error">
                    <div class="verdict-title">❌ Mismatch</div>
                    <div class="verdict-desc">The image does not appear to match the description.</div>
                </div>
                """
                bar_color = "#ef4444" # Red
            
            # Display The Card
            st.markdown(html_card, unsafe_allow_html=True)
            
            # --- VISUAL BAR (No Numbers) ---
            # We normalize the score (0.0 to 0.4) to a percentage (0% to 100%) purely for the CSS width
            # 0.4 is usually a max score for CLIP raw text/image pairs
            display_width = min(max((score / 0.35) * 100, 5), 100) 
            
            st.markdown(f"""
            <div style="margin-bottom: 30px;">
                <div class="bar-labels">
                    <span>Low</span>
                    <span>Medium</span>
                    <span>High</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: {display_width}%; background-color: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- DETAILS SECTION ---
            st.markdown("#### 🧐 Analysis Details")
            st.info(f"**AI Vision:** The system identified *\"{ai_caption}\"*.")
            st.caption(f"**Your Text:** *\"{user_text}\"*")