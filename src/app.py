import streamlit as st
from PIL import Image
import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)

st.set_page_config(page_title="Text–Image Consistency Checker")

st.title("🖼️ Text–Image Semantic Consistency Checker")

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
    st.image(image, caption="Uploaded Image", width=300)

    # BLIP caption
    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=30)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # CLIP similarity
    clip_inputs = clip_processor(
        text=[user_text, caption],
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

    # Verdict
    if final_score >= 0.35:
        verdict = "✅ CONSISTENT"
    elif final_score >= 0.20:
        verdict = "⚠️ PARTIALLY CONSISTENT"
    else:
        verdict = "❌ INCONSISTENT"

    st.subheader("Results")
    st.write("**Image Caption:**", caption)
    st.write("**Final Score:**", round(final_score, 3))
    st.write("**Verdict:**", verdict)
