#!/usr/bin/env python3
"""
Voice Chat Agent using Silero VAD, Whisper, Qwen2.5-VL, and Piper TTS
Author: Joshua D. Boyd

"""

import asyncio
import pyaudio
import wave
import torch
import torchaudio
import numpy as np
import tempfile
import os
import io
import argparse
import time

# Disable tokenizers parallelism warning in multiprocessing context
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import platform
import subprocess
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import mlx_whisper
from piper import PiperVoice
from mlx_lm import load, generate
import strip_markdown
import pyttsx3


@dataclass
class AudioChunk:
    data: np.ndarray
    timestamp: float
    is_speech: bool


class VoiceChatAgent:
    def __init__(
        self,
        whisper_model: str = "base",
        qwen_model_name: str = "mlx-community/Qwen3-1.7B-6bit",
        piper_model_path: str = "./piper_models/en_US-hfc_female-medium.onnx",
        piper_config_path: str = "./piper_models/en_US-hfc_female-medium.onnx.json",
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        speech_threshold: float = 0.5,
        silence_duration: float = 1.0,
        playback_audio: bool = False,
        max_tokens: int = 512,
    ):
        """
        Initialize the voice chat agent.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            qwen_model_name: Qwen model name from Hugging Face
            piper_model_path: Path to Piper TTS model (.onnx file)
            piper_config_path: Path to Piper TTS config (.json file)
            sample_rate: Audio sample rate (16kHz recommended)
            chunk_duration: Duration of each audio chunk in seconds
            speech_threshold: VAD threshold (0.0-1.0)
            silence_duration: Seconds of silence before processing speech
            playback_audio: Whether to play back recorded audio before transcription
            max_tokens: Maximum tokens for Qwen response
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        # VAD expects specific chunk sizes: 512 samples for 16kHz, 256 for 8kHz
        self.vad_chunk_size = 512 if sample_rate == 16000 else 256
        self.speech_threshold = speech_threshold
        self.silence_duration = silence_duration
        self.piper_model_path = piper_model_path
        self.piper_config_path = piper_config_path
        self.playback_audio = playback_audio
        self.max_tokens = max_tokens
        self.qwen_model_name = qwen_model_name

        # Initialize components
        self._init_vad()
        self._init_whisper(whisper_model)
        self._init_qwen()
        self._init_tts()
        self._init_audio()

        # State management
        self.audio_buffer = []
        self.is_recording = False
        self.silence_counter = 0
        self.conversation_history = []
        self.is_speaking = False  # Flag to prevent feedback during TTS

        # Async queues
        self.audio_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

        print("🎤 Voice Chat Agent initialized!")
        print(f"Using Whisper model: {whisper_model}")
        print(f"Using Qwen model: {qwen_model_name}")
        if platform.system() == "Darwin":
            print("Using Apple TTS with Samantha voice")
        else:
            print(f"Using Piper model: {piper_model_path}")

    def _init_vad(self):
        """Initialize Silero VAD model."""
        print("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.vad_model = model
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils
        print("✅ Silero VAD loaded")

    def _init_whisper(self, model_name: str):
        """Initialize Whisper STT model."""
        # Use MLX-optimized 4-bit quantized model for better performance
        hf_model_name = "mlx-community/whisper-medium.en-mlx-4bit"
        print(f"Loading Whisper model: {model_name} -> {hf_model_name}...")
        self.whisper_model = mlx_whisper.load_models.load_model(hf_model_name)
        print("✅ Whisper loaded")

    def _init_qwen(self):
        """Initialize Qwen3 model using MLX."""
        print(f"Loading Qwen model: {self.qwen_model_name}...")
        print("Using MLX framework for Apple Silicon optimization")

        # Load model and tokenizer using MLX-LM
        self.qwen_model, self.qwen_tokenizer = load(self.qwen_model_name)

        # Initialize conversation with system message
        self.system_message = """You are a helpful voice assistant. Keep your responses very concise and conversational for speech. Avoid lists or complex formatting since this will be read aloud. Aim for responses under 50 words unless specifically asked for detailed information. Be direct and to the point."""

        self.conversation_history = []
        print("✅ Qwen model loaded")

    def _init_tts(self):
        """Initialize TTS model based on platform."""
        if platform.system() == "Darwin":
            # On macOS, we use Apple's TTS via pyttsx3
            print("Initializing Apple TTS with pyttsx3...")
            self.tts_engine = pyttsx3.init("nsss")  # Use NSSpeechSynthesizer on macOS

            # Set voice to Samantha (female voice)
            voices = self.tts_engine.getProperty("voices")
            for voice in voices:
                if "samantha" in voice.name.lower():
                    self.tts_engine.setProperty("voice", voice.id)
                    break

            # Set speech rate and volume
            self.tts_engine.setProperty("rate", 200)  # Adjust speaking rate
            self.tts_engine.setProperty("volume", 1.0)  # Max volume

            print("✅ Apple TTS initialized with pyttsx3")
            self.piper_voice = None  # Not needed on macOS
        else:
            # On other platforms, use Piper TTS
            self.tts_engine = None
            self._init_piper()

    def _init_piper(self):
        """Initialize Piper TTS model."""
        print("Loading Piper TTS model...")
        try:
            self.piper_voice = PiperVoice.load(
                self.piper_model_path,
                config_path=self.piper_config_path,
                use_cuda=torch.cuda.is_available(),
            )
            print("✅ Piper TTS loaded")
        except Exception as e:
            print(f"❌ Failed to load Piper TTS: {e}")
            print("Make sure you have both the .onnx model file and .json config file")
            raise

    def _init_audio(self):
        """Initialize PyAudio for microphone input."""
        self.p = pyaudio.PyAudio()
        print("✅ Audio system initialized")

    def detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """Use Silero VAD to detect speech in audio chunk."""
        # Silero VAD expects exactly 512 samples for 16kHz or 256 for 8kHz
        expected_samples = 512 if self.sample_rate == 16000 else 256

        # If chunk is too long, take the first expected_samples
        if len(audio_chunk) > expected_samples:
            audio_chunk = audio_chunk[:expected_samples]
        # If chunk is too short, pad with zeros
        elif len(audio_chunk) < expected_samples:
            audio_chunk = np.pad(
                audio_chunk, (0, expected_samples - len(audio_chunk)), "constant"
            )

        # Convert to tensor and ensure correct format
        audio_tensor = torch.FloatTensor(audio_chunk).unsqueeze(
            0
        )  # Add batch dimension

        # Get speech probability
        with torch.no_grad():
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

        return speech_prob > self.speech_threshold

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using MLX Whisper."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Write audio to temp file
                with wave.open(tmp_file.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    # Convert float32 to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())

                # Transcribe using mlx_whisper.transcribe function
                result = mlx_whisper.transcribe(
                    tmp_file.name,
                    path_or_hf_repo="mlx-community/whisper-medium.en-mlx-4bit",
                )
                text = result["text"].strip()

                # Clean up
                os.unlink(tmp_file.name)

                return text

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

    def format_conversation_for_qwen(self, user_text: str) -> List[dict]:
        """Format the conversation history for Qwen3 chat format."""
        messages = []

        # Add system message
        messages.append({"role": "system", "content": self.system_message})

        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)

        # Add current user message with /no_think prefix
        messages.append({"role": "user", "content": f"/no_think {user_text}"})

        return messages

    async def get_qwen_response(self, user_text: str) -> str:
        """Get response from Qwen model using MLX."""
        try:
            # Format the conversation for Qwen3
            messages = self.format_conversation_for_qwen(user_text)

            # Prepare the prompt using tokenizer chat template if available
            if self.qwen_tokenizer.chat_template is not None:
                prompt = self.qwen_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback to simple prompt format
                prompt = f"System: {self.system_message}\n\n"
                for msg in self.conversation_history:
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    prompt += f"{role}: {content}\n"
                prompt += f"User: /no_think {user_text}\nAssistant:"

            print(f"📊 Prompt length: {len(prompt)} characters")

            # Generate response using MLX-LM
            response_text = generate(
                self.qwen_model,
                self.qwen_tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                verbose=False,
            )

            # Clean up the response (remove the prompt if it's included)
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt) :].strip()

            # Add to conversation history (using simple format for history)
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )

            # Keep conversation history manageable (last 6 exchanges = 12 messages)
            if len(self.conversation_history) > 12:
                self.conversation_history = self.conversation_history[-12:]

            return response_text

        except Exception as e:
            print(f"❌ Qwen model error: {e}")
            import traceback

            traceback.print_exc()
            return "Sorry, I'm having trouble processing that right now."

    def clean_text_for_tts(self, text: str) -> str:
        """Clean text by removing markdown formatting that could interfere with TTS."""
        # Use strip-markdown to remove markdown formatting
        clean_text = strip_markdown.strip_markdown(text)

        # Additional cleaning for SSML safety - remove HTML-like tags that
        # might be interpreted as SSML (but preserve valid SSML tags)
        clean_text = re.sub(
            r"<(?!/?(?:speak|break|emphasis|phoneme|prosody|say-as|sub|voice)\b)[^>]*>",
            "",
            clean_text,
        )

        # Clean up extra whitespace
        clean_text = re.sub(r"\s+", " ", clean_text)
        clean_text = clean_text.strip()

        return clean_text

    def synthesize_speech(self, text: str) -> bool:
        """Convert text to speech using Apple TTS on macOS or Piper TTS on other platforms."""
        try:
            # Clean text to remove markdown and prevent SSML conflicts
            clean_text = self.clean_text_for_tts(text)

            if not clean_text.strip():
                print("⚠️  No text to synthesize after cleaning")
                return False

            print(f"🧹 Cleaned text: {clean_text}")

            # Check if we're on macOS and use Apple TTS
            if platform.system() == "Darwin":
                return self._synthesize_with_apple_tts(clean_text)
            else:
                return self._synthesize_with_piper_tts(clean_text)

        except Exception as e:
            print(f"❌ Speech synthesis error: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _synthesize_with_apple_tts(self, text: str) -> bool:
        """Use Apple's TTS API via pyttsx3."""
        try:
            # Set speaking flag to prevent audio feedback
            self.is_speaking = True

            # Use pyttsx3 to speak the text
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()  # Wait for speech to complete

            # Add brief delay to let audio echo dissipate before reactivating microphone
            time.sleep(0.5)

            # Clear speaking flag
            self.is_speaking = False

            return True

        except Exception as e:
            print(f"❌ Apple TTS error: {e}")
            self.is_speaking = False
            return False

    def _synthesize_with_piper_tts(self, text: str) -> bool:
        """Use Piper TTS for non-macOS platforms."""
        try:
            # Set speaking flag to prevent audio feedback
            self.is_speaking = True

            # Generate audio using Piper
            audio_bytes = b""
            for audio_chunk in self.piper_voice.synthesize(text):
                # AudioChunk objects have audio_int16_bytes attribute
                audio_bytes += audio_chunk.audio_int16_bytes

            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Write WAV header and audio data
                with wave.open(tmp_file.name, "wb") as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.piper_voice.config.sample_rate)
                    wav_file.writeframes(audio_bytes)

                # Play the audio
                self.play_audio_file(tmp_file.name)

                # Clean up
                os.unlink(tmp_file.name)

            # Add brief delay to let audio echo dissipate before reactivating microphone
            time.sleep(0.5)

            # Clear speaking flag
            self.is_speaking = False

            return True

        except Exception as e:
            print(f"❌ Piper TTS error: {e}")
            self.is_speaking = False
            return False

    def play_audio_file(self, file_path: str):
        """Play audio file using PyAudio."""
        try:
            with wave.open(file_path, "rb") as wav_file:
                # Get audio parameters
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()

                # Open audio stream
                stream = self.p.open(
                    format=self.p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                )

                # Play audio
                data = wav_file.readframes(1024)
                while data:
                    stream.write(data)
                    data = wav_file.readframes(1024)

                stream.stop_stream()
                stream.close()

        except Exception as e:
            print(f"❌ Audio playback error: {e}")

    async def audio_stream_generator(self) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio chunks from microphone asynchronously."""
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        stream.start_stream()

        try:
            while True:
                # Read audio data
                audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
                yield audio_chunk

                # Small delay to prevent overwhelming the CPU
                await asyncio.sleep(0.01)

        except Exception as e:
            print(f"❌ Audio stream error: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    async def process_audio_stream(self):
        """Process audio stream for VAD and speech recognition."""
        vad_buffer = np.array([])

        async for audio_chunk in self.audio_stream_generator():
            try:
                # Add to VAD buffer
                vad_buffer = np.concatenate([vad_buffer, audio_chunk])

                # Process VAD in smaller chunks if we have enough data
                while len(vad_buffer) >= self.vad_chunk_size:
                    # Extract chunk for VAD
                    vad_chunk = vad_buffer[: self.vad_chunk_size]
                    vad_buffer = vad_buffer[self.vad_chunk_size :]

                    # Detect speech only if we're not currently speaking (to prevent feedback)
                    if not self.is_speaking:
                        has_speech = self.detect_speech(vad_chunk)

                        if has_speech:
                            if not self.is_recording:
                                print("🗣️  Speech detected, recording...")
                                self.is_recording = True
                                self.audio_buffer = []

                            # Add the VAD chunk (not the full audio_chunk) to buffer
                            self.audio_buffer.extend(vad_chunk)
                            self.silence_counter = 0

                        else:
                            if self.is_recording:
                                self.silence_counter += 1
                                # Add the VAD chunk to buffer during recording
                                self.audio_buffer.extend(vad_chunk)

                                # Check if we've had enough silence to stop recording
                                # Use VAD chunk duration instead of full chunk duration
                                vad_chunk_duration = (
                                    self.vad_chunk_size / self.sample_rate
                                )
                                silence_duration = (
                                    self.silence_counter * vad_chunk_duration
                                )
                                if silence_duration >= self.silence_duration:
                                    print("🤫 Silence detected, processing speech...")
                                    await self._process_speech()
                                    self.is_recording = False
                                    self.silence_counter = 0
                    else:
                        # If we're speaking, skip all audio processing to prevent feedback
                        pass

            except Exception as e:
                print(f"❌ Audio processing error: {e}")
                import traceback

                traceback.print_exc()

    async def _process_speech(self):
        """Process recorded speech through the full pipeline."""
        if not self.audio_buffer:
            return

        try:
            # Convert buffer to numpy array
            audio_array = np.array(self.audio_buffer, dtype=np.float32)

            # Play back recorded audio if requested
            if self.playback_audio:
                print("🔊 Playing back recorded audio...")
                self._playback_recorded_audio(audio_array)
                print("✅ Playback complete")

            # Transcribe
            print("🎯 Transcribing...")
            text = self.transcribe_audio(audio_array)

            if text:
                print(f"📝 Transcription: {text}")
                print(f"👤 You said: {text}")

                # Queue for Qwen processing
                await self.response_queue.put(text)
            else:
                print("❓ Could not understand speech")

        except Exception as e:
            print(f"❌ Speech processing error: {e}")

    def _playback_recorded_audio(self, audio_data: np.ndarray):
        """Play back the recorded audio data."""
        try:
            # Create temporary file for playback
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Write audio to temp file
                with wave.open(tmp_file.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    # Convert float32 to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())

                # Play the audio
                self.play_audio_file(tmp_file.name)

                # Clean up
                os.unlink(tmp_file.name)

        except Exception as e:
            print(f"❌ Audio playback error: {e}")

    async def handle_responses(self):
        """Handle Qwen responses and TTS."""
        while True:
            try:
                # Wait for user text from speech processing
                user_text = await self.response_queue.get()

                print("🤖 Getting Qwen response...")
                response = await self.get_qwen_response(user_text)
                print(f"🤖 Qwen: {response}")

                print("🔊 Converting to speech...")
                self.synthesize_speech(response)
                print("✅ Response complete!")
                print("🎙️  Listening for more speech...\n")

            except Exception as e:
                print(f"❌ Response handling error: {e}")
                import traceback

                traceback.print_exc()

    async def start_listening(self):
        """Start continuous audio capture and processing."""
        print("🎙️  Starting to listen... (Press Ctrl+C to stop)")

        # Run audio processing and response handling concurrently
        try:
            await asyncio.gather(self.process_audio_stream(), self.handle_responses())
        except KeyboardInterrupt:
            print("\n🛑 Stopping voice chat...")
        finally:
            self.p.terminate()


async def main():
    """Main function to run the voice chat agent."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Voice Chat Agent with Qwen2.5")
    parser.add_argument(
        "--playback-audio",
        action="store_true",
        help="Play back recorded audio before transcription (useful for debugging)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--qwen-model",
        default="mlx-community/Qwen3-1.7B-6bit",
        help="Qwen model name from Hugging Face (default: mlx-community/Qwen3-1.7B-6bit)",
    )
    parser.add_argument(
        "--speech-threshold",
        type=float,
        default=0.5,
        help="VAD speech detection threshold 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=1.0,
        help="Seconds of silence before processing speech (default: 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for Qwen response (default: 512)",
    )

    args = parser.parse_args()

    # Paths to Piper model files (download from https://github.com/rhasspy/piper/releases)
    PIPER_MODEL_PATH = "./piper_models/en_US-hfc_female-medium.onnx"
    PIPER_CONFIG_PATH = "./piper_models/en_US-hfc_female-medium.onnx.json"

    # Check for Piper models only if not on macOS
    if platform.system() != "Darwin":
        if not os.path.exists(PIPER_MODEL_PATH):
            print(f"❌ Piper model not found at {PIPER_MODEL_PATH}")
            print("Download models from: https://github.com/rhasspy/piper/releases")
            return

        if not os.path.exists(PIPER_CONFIG_PATH):
            print(f"❌ Piper config not found at {PIPER_CONFIG_PATH}")
            print("Make sure to download both .onnx and .json files")
            return

    # Show configuration
    if args.playback_audio:
        print(
            "🔊 Audio playback enabled - recorded audio will be played back before transcription"
        )

    print(f"🧠 Using Qwen model: {args.qwen_model}")
    print(f"🎯 Max tokens: {args.max_tokens}")

    # Create and start the voice chat agent
    agent = VoiceChatAgent(
        whisper_model=args.whisper_model,
        qwen_model_name=args.qwen_model,
        piper_model_path=PIPER_MODEL_PATH,
        piper_config_path=PIPER_CONFIG_PATH,
        speech_threshold=args.speech_threshold,
        silence_duration=args.silence_duration,
        playback_audio=args.playback_audio,
        max_tokens=args.max_tokens,
    )

    await agent.start_listening()


if __name__ == "__main__":
    asyncio.run(main())
