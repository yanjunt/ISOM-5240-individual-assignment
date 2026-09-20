import io
import re

import streamlit as st
from PIL import Image
from transformers import (
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline,
)
from gtts import gTTS



# MODEL LOADING FUNCTIONS



def load_captioner():
    # Load the pre-trained BLIP image-captioning model from Hugging Face.
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model



def load_story_generator():
    # Load the Hugging Face text-generation pipeline used to create the story.
    generator = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
    )

    # Load the tokenizer so that the model's chat template can be used.
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    return generator, tokenizer



# IMAGE CAPTIONING


def generate_caption(image, processor, model):
    # Convert the uploaded image into the format required by BLIP.
    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    # Generate a text description of the uploaded image.
    outputs = model.generate(**inputs)

    # Convert the model output into readable text.
    caption = processor.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return caption


# STORY GENERATION


def truncate_to_sentence(text, max_words=100, min_words=50):
    # Split the generated story into individual words.
    words = text.split()

    # Return the original story if it is already within the maximum length.
    if len(words) <= max_words:
        return text

    # Keep only the first 100 words when the story is too long.
    truncated_text = " ".join(words[:max_words])

    # Look for the last complete sentence within the 100-word limit.
    sentence_endings = list(re.finditer(r"[.!?]", truncated_text))

    if sentence_endings:
        last_ending = sentence_endings[-1]
        text = truncated_text[:last_ending.end()].strip()
    else:
        text = truncated_text.strip() + "."

    # If truncation makes the story shorter than 50 words,
    # add a short child-friendly ending.
    if len(text.split()) < min_words:
        text += " What a happy day."

    return text


def caption_to_story(caption, generator, tokenizer):
    # Create an instruction that tells the language model exactly what type of story it should generate.
    messages = [
        {
            "role": "user",
            "content": (
                "Write a short, happy story for a child aged 3 to 10. "
                f"The story must be based ONLY on this image description: {caption}. "
                "Requirements: 50 to 100 words. "
                "Use simple English. "
                "Include a clear beginning, middle, and ending. "
                "Make the story warm, playful, and positive. "
                "Do not mention unrelated things. "
                "Do not repeat sentences. "
                "Output only the story."
            ),
        }
    ]

    # Format the instruction using the model's chat template.
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Generate the children's story with controlled sampling settings.The temprature has been tested and selected. And penalty added to avoid repetition.
    result = generator(
        formatted,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        return_full_text=False,
    )

    # Extract and clean the generated story.
    story = result[0]["generated_text"].strip()

    # Keep the story within the assignment's 50-100 word requirement.
    story = truncate_to_sentence(
        story,
        max_words=100,
        min_words=50,
    )

    return story



# TEXT-TO-SPEECH


def text_to_audio(text, lang="en"):
    # Convert the generated story into speech using Google Text-to-Speech.
    tts = gTTS(
        text=text,
        lang=lang,
    )

    # Store the generated MP3 audio in memory instead of creating
    # a permanent local file.
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return audio_buffer



# STREAMLIT USER INTERFACE


def main():
    # Configure the Streamlit page title and browser icon.
    st.set_page_config(
        page_title="Kids Picture Storyteller",
        page_icon="📖",
    )

    # Display the main application title and instructions.
    st.title("📖 Kids Picture Storyteller")
    st.write(
        "Upload a picture. The app creates a 50-100 words children's story and reads it aloud."
    )

    # Allow users to upload JPG, JPEG, or PNG images.
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )

    # Continue only after the user has uploaded an image.
    if uploaded_file is not None:
        # Open the uploaded image and convert it to RGB format.
        image = Image.open(uploaded_file).convert("RGB")

        # Display the uploaded image in the Streamlit interface.
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

        # Confirm that the image has been uploaded successfully.
        st.success("✅ Image uploaded successfully!")

        # Start the complete storytelling pipeline when the button is clicked.
        if st.button("Make my story!"):
            st.info(
                "⏳ The first run may take several minutes while the AI models load. "
                "Please do not refresh the page."
            )

            # Create a status placeholder for progress messages.
            status = st.empty()

            # --------------------------------------------------------
            # STEP 1: IMAGE PROCESSING AND CAPTIONING
            # --------------------------------------------------------
            status.info(
                "Step 1/3: Loading BLIP and generating the image caption..."
            )

            with st.spinner("Generating image caption..."):
                processor, caption_model = load_captioner()
                caption = generate_caption(
                    image,
                    processor,
                    caption_model,
                )

            # Display the caption generated from the uploaded image.
            st.subheader("Image Caption")
            st.write(caption)

            # --------------------------------------------------------
            # STEP 2: STORY GENERATION
            # --------------------------------------------------------
            status.info(
                "Step 2/3: Loading the story-generation model and writing the story..."
            )

            with st.spinner("Writing your story..."):
                story_generator, story_tokenizer = load_story_generator()
                story = caption_to_story(
                    caption,
                    story_generator,
                    story_tokenizer,
                )

            # Display the generated children's story.
            st.subheader("Story")
            st.write(story)

            # Display the number of words generated for transparency.
            st.caption(
                f"Word count: {len(story.split())}"
            )

            # --------------------------------------------------------
            # STEP 3: TEXT-TO-SPEECH
            # --------------------------------------------------------
            status.info(
                "Step 3/3: Converting the story into audio..."
            )

            with st.spinner("Generating audio..."):
                audio_buffer = text_to_audio(
                    story,
                    lang="en",
                )

            # Provide an audio player so children can listen to the story.
            st.subheader("🔊 Listen to the Story")
            st.audio(
                audio_buffer,
                format="audio/mp3",
            )

            # Inform the user that all three stages have completed.
            status.success("✅ All done! Enjoy your story!")


# Run the Streamlit application.
if __name__ == "__main__":
    main()
