import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, render_template_string, send_from_directory, url_for
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)



# Folder to store generated audio
OUTPUT_DIR = Path("static/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing!")

client = OpenAI(api_key=api_key)

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=3000
)

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a professional storyteller, narrator, and singer.

    Generate a detailed and complete narration about the topic.
    Make it natural, clear, and suitable for audio.
    No background noise.
    Only deliver the content strictly.
    It should deliver the full content / full story.
    The story/concept must be complete and should not stop midway.

    Topic:
    {topic}
    """
)

chain = prompt | llm


def generate_text(topic: str) -> str:
    response = chain.invoke({"topic": topic})
    return response.content.strip()


def text_to_audio(text: str, output_file: str):
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="nova",   # alloy, verse, aria, sage, ember, nova
        input=text
    ) as response:
        response.stream_to_file(output_file)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Story to Audio Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f8f9fa;
        }
        h1 {
            color: #222;
        }
        form {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            margin-top: 8px;
            margin-bottom: 16px;
            border: 1px solid #ccc;
            border-radius: 8px;
        }
        button {
            padding: 12px 18px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        button:hover {
            background: #0056b3;
        }
        .result {
            margin-top: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .story-box {
            white-space: pre-wrap;
            line-height: 1.6;
            background: #f3f3f3;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        audio {
            margin-top: 15px;
            width: 100%;
        }
    </style>
</head>
<body>
    <h1>Story / Concept to Audio Generator</h1>

    <form method="POST">
        <label><strong>Enter a topic:</strong></label>
        <input type="text" name="topic" placeholder="e.g. Life of APJ Abdul Kalam" required>
        <button type="submit">Generate</button>
    </form>

    {% if error %}
        <div class="result">
            <h3>Error</h3>
            <p>{{ error }}</p>
        </div>
    {% endif %}

    {% if topic and generated_text and audio_file %}
        <div class="result">
            <h2>Topic: {{ topic }}</h2>

            <h3>Generated Narration</h3>
            <div class="story-box">{{ generated_text }}</div>

            <h3>Audio Output</h3>
            <audio controls>
                <source src="{{ audio_file }}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>

            <p>
                <a href="{{ audio_file }}" download>Download Audio</a>
            </p>
        </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        topic = request.form.get("topic", "").strip()

        if not topic:
            return render_template_string(HTML_PAGE, error="Please enter a valid topic.")

        try:
            generated_text = generate_text(topic)

            safe_filename = topic.replace(" ", "_").replace("/", "_").replace("\\", "_")
            output_path = OUTPUT_DIR / f"{safe_filename}.mp3"

            text_to_audio(generated_text, str(output_path))

            audio_url = url_for("static", filename=f"audio/{safe_filename}.mp3")

            return render_template_string(
                HTML_PAGE,
                topic=topic,
                generated_text=generated_text,
                audio_file=audio_url
            )

        except Exception as e:
            return render_template_string(HTML_PAGE, error=str(e))

    return render_template_string(HTML_PAGE)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)