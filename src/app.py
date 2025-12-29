import streamlit as st
from PIL import Image
import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
from collections import Counter

st.set_page_config(page_title="Text–Image Consistency Checker", layout="wide")

st.title("🖼️ Text–Image Semantic Consistency Checker")
st.markdown(
    "Upload an image and enter a description to check if the text matches the image content."
)

# Load models (once)
@st.cache_resource
def load_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return clip_model, clip_processor, blip_model, blip_processor

clip_model, clip_processor, blip_model, blip_processor = load_models()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
user_text = st.text_input("Enter image description")

if uploaded_image and user_text:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    # ---------- STEP 1: GENERATE MULTIPLE CAPTIONS ----------
    inputs = blip_processor(image, return_tensors="pt")
    num_captions = 3
    captions = []

    with torch.no_grad():
        for _ in range(num_captions):
            out = blip_model.generate(**inputs, max_new_tokens=30)
            caption = blip_processor.decode(out[0], skip_special_tokens=True).lower()
            captions.append(caption)

    # ---------- STEP 2: REFINE CAPTION (KEEP COMMON WORDS) ----------
    words_list = [caption.split() for caption in captions]
    word_counter = Counter()
    for words in words_list:
        word_counter.update(words)

    # Keep words that appear in at least 2 captions, preserving order
    refined_words = []
    for caption in captions:
        for word in caption.split():
            if word_counter[word] >= 2 and word not in refined_words:
                refined_words.append(word)

    refined_caption = " ".join(refined_words)

    st.markdown("### 🔹 Generated Captions")
    st.write(captions)
    st.markdown("### 🔹 Refined Caption")
    st.success(refined_caption)

    # ---------- STEP 3: CLIP SIMILARITY ----------
    clip_inputs = clip_processor(
        text=[user_text, refined_caption],
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = clip_model(**clip_inputs)

    img_emb = outputs.image_embeds
    txt_emb = outputs.text_embeds

    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    score_user = (img_emb @ txt_emb[0].T).item()
    score_caption = (img_emb @ txt_emb[1].T).item()
    final_score = (score_user + score_caption) / 2

    # ---------- STEP 4: VERDICT ----------
    if final_score >= 0.35:
        verdict = "✅ CONSISTENT"
        color = "green"
    elif final_score >= 0.20:
        verdict = "⚠️ PARTIALLY CONSISTENT"
        color = "orange"
    else:
        verdict = "❌ INCONSISTENT"
        color = "red"

    # ---------- DISPLAY RESULTS WITH EFFECTS ----------
    st.markdown("### 📝 Final Evaluation")
    st.markdown(f"**Final Score:** {round(final_score, 3)}")
    st.markdown(f"**Verdict:** <span style='color:{color}; font-size:24px'>{verdict}</span>", unsafe_allow_html=True)
