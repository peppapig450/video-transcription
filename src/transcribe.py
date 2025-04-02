import argparse
import logging
import os
import sys
import tempfile
from datetime import timedelta

import torch
import whisperx
import whisperx.types
from dotenv import load_dotenv
from ffmpeg import FFmpeg, FFmpegError
from whisperx.types import SingleAlignedSegment

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Supported WhisperX models
SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


def extract_audio(video_path: str) -> str:
    """
    Extract audio from a video file using FFmpeg and save it as a temporary WAV file.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        str: Path to the extracted audio file.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ffmpeg.Error: If FFmpeg fails to extract the audio.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio_output_path = temp_audio.name

    try:
        ffmpeg_instance = FFmpeg()
        ffmpeg_instance.input(video_path).output(
            audio_output_path,
            format="wav",  # Output format
            acodec="pcm_s16le",  # Audio codec: 16-bit PCM
            ar=16000,  # Audio sample rate: 16kHz
        ).execute()

        logging.info("Audio extracted to %s", audio_output_path)
        return audio_output_path
    except FFmpegError:
        logging.exception("Error extracting audio: ")
        raise
    except Exception:
        logging.exception("Unexpected error: ")
        raise


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    output_text: str = "transcript.txt",
    output_srt: str = "transcript.srt",
    hf_token: str | None = None,
) -> tuple[str, list[SingleAlignedSegment]]:
    """
    Transcribe audio using WhisperX with speaker diarization and save to text and SRT files.

    Args:
        audio_path (str): Path to the input audio file.
        model_name (str): Name of the WhisperX model to use. Defaults to "base".
        output_text (str): Path to save the plain text transcript. Defaults to "transcript.txt".
        output_srt (str): Path to save the SRT subtitle file. Defaults to "transcript.srt".
        hf_token (str | None): Hugging Face token for diarization models. Defaults to None (loaded from .env).

    Returns:
        tuple[str, list[SingleAlignedSegment]]: The transcript text and list of segments with timing and speaker info.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the model name is not supported.
        Exception: For any other errors during transcription.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    compute_type = (
        "float16" if device == "cuda" else "float32"
    )  # Use int8 for lower memory

    # Use token from .env if not provided on the command-line
    hf_token = hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        logging.warning(
            "No Hugging Face token provided (via --hf_token or .env); skipping speaker diarization."
        )

    try:
        # Load WhisperX model
        logging.info("Loading WhisperX model '%s' on %s...", model_name, device)
        model = whisperx.load_model(
            "large-v2", device, compute_type=compute_type, download_root="models"
        )

        # Load audio
        audio = whisperx.load_audio(audio_path)

        # Transcribe audio
        logging.info("Transcribing audio...")
        result = model.transcribe(audio, batch_size=batch_size, print_progress=True)

        # Align segments for precise timestamps
        logging.info("Aligning segments...")
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        # Perform speaker diarization
        if hf_token:
            logging.info("Performing speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        # Combine text for plain transcript
        transcript = " ".join(segment["text"] for segment in result["segments"])

        # Save plain text transcript
        with open(output_text, "w", encoding="utf-8") as file:
            file.write(transcript)
        logging.info("Plain text transcript saved to %s", output_text)

        # Save SRT with speaker labels
        with open(output_srt, "w", encoding="utf-8") as file:
            for i, segment in enumerate(result["segments"], start=1):
                start_time = timedelta(seconds=segment["start"])
                end_time = timedelta(seconds=segment["end"])
                start_str = str(start_time).replace(".", ",").zfill(12)
                end_str = str(end_time).replace(".", ",").zfill(12)

                speaker = segment.get("speaker", "Unknown") if hf_token else "Unknown"
                file.write(
                    f"{i}\n{start_str} --> {end_str}\nSpeaker {speaker}: {segment['text'].strip()}\n\n"
                )
        logging.info(f"SRT transcript with speaker labels saved to {output_srt}")

        return transcript, result["segments"]
    except Exception:
        logging.exception("Error during transcription")
        raise


def video_to_transcript(
    video_path: str,
    model_name: str = "base",
    output_text: str = "transcript.txt",
    output_srt: str = "transcript.srt",
    keep_audio: bool = False,
    hf_token: str | None = None,
) -> None:
    """
    Extract audio from a video and transcribe it using WhisperX with speaker diarization.

    Args:
        video_path (str): Path to the input video file.
        model_name (str): Name of the WhisperX model to use. Defaults to "base".
        output_text (str): Path to save the plain text transcript. Defaults to "transcript.txt".
        output_srt (str): Path to save the SRT subtitle file. Defaults to "transcript.srt".
        keep_audio (bool): Whether to keep the temporary audio file. Defaults to False.
        hf_token (str | None): Hugging Face token for diarization. Defaults to None (loaded from .env).
    """
    audio_path = extract_audio(video_path)
    try:
        transcript, segments = transcribe_audio(
            audio_path, model_name, output_text, output_srt, hf_token
        )
    finally:
        if not keep_audio and os.path.exists(audio_path):
            os.remove(audio_path)
            logging.info("Cleaned up temporary file: %s", audio_path)

    logging.info("\nTranscript Preview:")
    preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
    logging.info(preview)


def main():
    """Parse command-line arguments and run the transcription process."""
    parser = argparse.ArgumentParser(
        description="Transcribe video to text and SRT using WhisperX with speaker diarization."
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument(
        "--model",
        default="base",
        choices=SUPPORTED_MODELS,
        help="WhisperX model to use (default: base).",
    )
    parser.add_argument(
        "--output_text",
        default="transcript.txt",
        help="Output plain text transcript file (default: transcript.txt).",
    )
    parser.add_argument(
        "--output_srt",
        default="transcript.srt",
        help="Output SRT subtitle file (default: transcript.srt).",
    )
    parser.add_argument(
        "--keep_audio",
        action="store_true",
        help="Keep the temporary audio file after transcription.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token for speaker diarization (overrides .env if provided).",
    )

    args = parser.parse_args()

    try:
        video_to_transcript(
            args.video_path,
            model_name=args.model,
            output_text=args.output_text,
            output_srt=args.output_srt,
            keep_audio=args.keep_audio,
            hf_token=args.hf_token,
        )
    except (FileNotFoundError, ValueError, FFmpegError, Exception):
        logging.exception("Failed to process video")
        sys.exit(1)


if __name__ == "__main__":
    main()
