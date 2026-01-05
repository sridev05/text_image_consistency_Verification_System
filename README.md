# Text–Image Semantic Consistency Verification System

## 📌 Project Overview

With the rise of multimodal AI, combining images and text is becoming ubiquitous in applications like image captioning, text-to-image generation, multimedia search, and content moderation. However, current systems struggle to verify whether a given image truly aligns with its textual description.

This project develops an automated Text–Image Semantic Consistency Verification System that:

- Understands the visual content of images.
- Interprets the semantic meaning of textual prompts.
- Computes quantitative similarity scores to evaluate alignment.
- Provides interpretable consistency verdicts for the user.

By leveraging state-of-the-art vision–language models, the system can detect mislabeling, misleading descriptions, and errors in AI-generated images, offering a scalable and objective solution.

---

## 🛠 Features

- **Image Captioning:** Automatically generates descriptive captions from input images.  
- **Semantic Similarity Scoring:** Measures how closely the image content aligns with a given textual description.  
- **Consistency Verdict:** Outputs whether the image and text are semantically consistent, along with a similarity score.  
- **Cross-Modal Analysis:** Bridges visual and textual data to detect inconsistencies that single-modality models cannot.  

---

## ⚡ Use Cases

- **AI-Generated Image Verification:** Check if text-to-image models generate accurate results.  
- **Content Moderation:** Detect misleading or incorrectly labeled media.  
- **Search & Retrieval:** Improve multimedia search by filtering inconsistent results.  
- **Dataset Validation:** Automatically ensure large image-text datasets maintain quality alignment.  

---

## 🖥 Tech Stack

- **Python** – Core programming language.  
- **PyTorch / TensorFlow** – For vision–language models.  
- **Transformers (Hugging Face)** – For pre-trained multimodal models like CLIP.  
- **OpenCV / PIL** – Image handling and preprocessing.  
- **Flask / FastAPI** – Optional backend for serving the system.  

---
<img width="1365" height="573" alt="image" src="https://github.com/user-attachments/assets/d555472c-f79c-45d1-b3f0-0fff97b81318" />

## How It Works 
1. **Input:** User provides an image and a textual description.  
2. **Caption Generation:** The system generates a caption describing the image.  
3. **Embedding Extraction:** Convert both the generated caption and input text into semantic embeddings.  
4. **Similarity Computation:** Calculate a similarity score between the embeddings.  
5. **Verdict:** Output a consistency score and an interpretable alignment result (e.g., `Consistent` or `Inconsistent`).  
