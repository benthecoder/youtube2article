import argparse
import os
from subprocess import run

import yt_dlp
from openai import OpenAI

client = OpenAI(api_key="")


SYSTEM_PROMPT = """
Your task is to deliver an in-depth analysis of video transcripts, offering nuanced insights that are easily digestible. Focus on a detailed exploration of the content, with particular emphasis on explaining terminology and providing a thorough section-by-section summary.
"""

USER_PROMPT_TEMPLATE = """
Your task is to provide an in-depth analysis of a provided video transcript, structured to both inform and engage readers. Your narrative should unfold with clarity and insight, reflecting the style of a Paul Graham essay. Follow these major headings for organization:

# Intro
Begin with a narrative introduction that captivates the reader, setting the stage for an engaging exploration of bilingualism. Start with an anecdote or a surprising fact to draw in the reader, then succinctly summarize the main themes and objectives of the video.

# ELI5
Immediately follow with an ELI5 (Explain Like I'm 5) section. Use simple language and analogies to make complex ideas accessible and engaging, ensuring clarity and simplicity.

# Terminologies
- List and define key terminologies mentioned in the video in bullet points. Provide comprehensive yet understandable definitions for someone not familiar with the subject matter. Ensure this section naturally transitions from the ELI5, enriching the reader's understanding without overwhelming them.

# Summary
Your summary should unfold as a detailed and engaging narrative essay, deeply exploring the content of the video. This section is the core of your analysis and should be both informative and thought-provoking. When crafting your summary, delve deeply into the video’s main themes. Provide a comprehensive analysis of each theme, backed by examples from the video and relevant research in the field. This section should read as a compelling essay, rich in detail and analysis, that not only informs the reader but also stimulates a deeper consideration of the topic's nuances and complexities. Strive for a narrative that is as enriching and engaging as it is enlightening. Please include headings and subheadings to organize your analysis effectively if needed. It should be as detailed and comprehensive as possible.

# Takeaways
- End with actionable takeaways in bullet points, offering practical advice or steps based on the video content. These should relate directly to the insights discussed in your essay and highlight their real-world relevance and impact.
\n\n\nText: {}:"""

# Constants
AUDIO_FORMAT = "mp3"
PREFERRED_QUALITY = "96"
MAX_FILESIZE = 25 * 1024 * 1024  # 25MB
FFMPEG_AUDIO_CHANNELS = "1"  # Mono
FFMPEG_BITRATE = "32k"


def download_audio_from_youtube(url):
    """Downloads audio from the given YouTube URL and returns the filename."""

    filename = None

    def my_hook(d):
        nonlocal filename
        if d["status"] == "finished":
            filename = d["filename"]

    ydl_opts = {
        "outtmpl": "%(title)s.%(ext)s",
        "format": "worstaudio",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": AUDIO_FORMAT,
                "preferredquality": PREFERRED_QUALITY,
            }
        ],
        "max_filesize": MAX_FILESIZE,
        "progress_hooks": [my_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Strip the extension from the filename to use it for further processing
    if filename:
        return filename.rsplit(".", 1)[0]


def convert_audio_to_mono(audio_filename):
    """Converts the downloaded audio file to mono format with lower bitrate."""
    command = [
        "ffmpeg",
        "-i",
        f"{audio_filename}.{AUDIO_FORMAT}",
        "-ac",
        FFMPEG_AUDIO_CHANNELS,
        "-ab",
        FFMPEG_BITRATE,
        "-y",
        f"{audio_filename}_mono.{AUDIO_FORMAT}",
    ]
    run(command)


def transcribe_audio(audio_filename):
    """Transcribes the given audio file using OpenAI's audio transcriptions."""

    with open(f"{audio_filename}_mono.{AUDIO_FORMAT}", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text"
        )
    return transcription


def summarize_transcript(transcript, system_prompt, user_prompt_template):
    """Summarizes the transcript into bullet points using OpenAI."""
    summarize_prompt = user_prompt_template.format(transcript)

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summarize_prompt},
        ],
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


def main():
    """Main function to parse arguments and orchestrate the summarization of a YouTube video."""
    parser = argparse.ArgumentParser(description="Summarize YouTube videos.")
    parser.add_argument(
        "url", type=str, help="The URL of the YouTube video to summarize."
    )
    args = parser.parse_args()

    try:
        url = args.url.replace("\\", "")
        audio_filename = download_audio_from_youtube(url)
        convert_audio_to_mono(audio_filename)
        transcript = transcribe_audio(audio_filename)
        with open(f"{audio_filename}.txt", "w") as f:
            f.write(transcript)
        summary = summarize_transcript(transcript, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)
        # save the summary to a file
        with open(f"{audio_filename}_summary.md", "w") as f:
            f.write(summary)
    finally:
        # Cleanup downloaded and processed files only if they exist
        if os.path.exists(f"{audio_filename}.{AUDIO_FORMAT}"):
            os.remove(f"{audio_filename}.{AUDIO_FORMAT}")
        if os.path.exists(f"{audio_filename}_mono.{AUDIO_FORMAT}"):
            os.remove(f"{audio_filename}_mono.{AUDIO_FORMAT}")


if __name__ == "__main__":
    main()
