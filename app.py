import re
import io
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS

@st.cache_resource
def load_captioner():
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return proc, model

@st.cache_resource
def load_story_gen():
    return pipeline("text-generation", model="gpt2")

def img2text(image: Image.Image) -> str:
    proc, model = load_captioner()
    inputs = proc(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    return proc.decode(out[0], skip_special_tokens=True).strip()

def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def trim_story(text: str, low=50, high=100) -> str:
    words = text.split()
    if count_words(text) > high:
        return " ".join(words[:high]).strip() + "."
    if count_words(text) < low:
        return text.strip() + " The children laughed, played, and went home happy. Everyone slept well and dreamed of more adventures."
    return text.strip()

def text2story(caption: str) -> str:
    gen = load_story_gen()
    prompt = (
        f"Write a simple, friendly children's story for ages 3 to 10 based on this picture: {caption}. "
        f"The story must be 50 to 100 words, with easy words and a happy ending:"
    )
    out = gen(prompt, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9)
    raw = out[0]["generated_text"]
    if prompt in raw:
        raw = raw.split(prompt, 1)[1]
    story = trim_story(raw)
    if count_words(story) < 50:
        story = trim_story(
            f"Once upon a time, {caption}. A kind child saw the scene and smiled. "
            "Friends came to play, they shared food, sang songs, and explored gently. "
            "When the sun set, everyone said goodbye and went home safely. The child slept happily."
        )
    return story

def text2audio(story: str, lang="en") -> io.BytesIO:
    buf = io.BytesIO()
    gTTS(text=story, lang=lang).write_to_fp(buf)
    buf.seek(0)
    return buf

def main():
    st.set_page_config(page_title="Kids Picture Storyteller", page_icon="📖")
    st.title("📖 Kids Picture Storyteller")
    st.write("Upload a picture. The app writes a 50–100 word story for children aged 3–10 and reads it aloud.")

    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)
        if st.button("Make my story"):
            with st.spinner("Reading the picture..."):
                caption = img2text(image)
            st.subheader("Image caption")
            st.write(caption)

            with st.spinner("Writing the story..."):
                story = text2story(caption)
            st.subheader("Story (50–100 words)")
            st.write(story)
            st.caption(f"Word count: {count_words(story)}")

            with st.spinner("Making audio..."):
                audio = text2audio(story, lang="en")
            st.subheader("Listen")
            st.audio(audio, format="audio/mp3")

if __name__ == "__main__":
    main()
