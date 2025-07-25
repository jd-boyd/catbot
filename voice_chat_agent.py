#!/usr/bin/env python3
"""
Voice Chat Agent using Silero VAD, Whisper, Claude Sonnet 4, and Piper TTS
Author: Joshua D. Boyd

Dependencies:
pip install torch torchaudio silero-vad openai-whisper anthropic langchain-anthropic langchain-core pyaudio wave numpy piper-tts
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
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import whisper
from piper import PiperVoice
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


@dataclass
class AudioChunk:
    data: np.ndarray
    timestamp: float
    is_speech: bool


class VoiceChatAgent:
    def __init__(
        self,
        anthropic_api_key: str,
        whisper_model: str = "base",
        piper_model_path: str = "./piper_models/en_US-hfc_female-medium.onnx",
        piper_config_path: str = "./piper_models/en_US-hfc_female-medium.onnx.json",
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        speech_threshold: float = 0.5,
        silence_duration: float = 1.0
    ):
        """
        Initialize the voice chat agent.

        Args:
            anthropic_api_key: Your Anthropic API key
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            piper_model_path: Path to Piper TTS model (.onnx file)
            piper_config_path: Path to Piper TTS config (.json file)
            sample_rate: Audio sample rate (16kHz recommended)
            chunk_duration: Duration of each audio chunk in seconds
            speech_threshold: VAD threshold (0.0-1.0)
            silence_duration: Seconds of silence before processing speech
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

        # Initialize components
        self._init_vad()
        self._init_whisper(whisper_model)
        self._init_claude(anthropic_api_key)
        self._init_piper()
        self._init_audio()

        # State management
        self.audio_buffer = []
        self.is_recording = False
        self.silence_counter = 0
        self.conversation_history = []

        # Async queues
        self.audio_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

        print("🎤 Voice Chat Agent initialized!")
        print(f"Using Whisper model: {whisper_model}")
        print(f"Using Piper model: {piper_model_path}")

    def _init_vad(self):
        """Initialize Silero VAD model."""
        print("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False
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
        print(f"Loading Whisper model: {model_name}...")
        self.whisper_model = whisper.load_model(model_name)
        print("✅ Whisper loaded")

    def _init_claude(self, api_key: str):
        """Initialize Claude LLM."""
        print("Initializing Claude API...")
        self.claude = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            max_tokens=1000,
            temperature=0.7,
        )

        # Set system message
        system_msg = SystemMessage(
            content="""
        You are a helpful voice assistant. Keep your responses conversational, 
        concise, and natural for speech. Avoid long lists or complex formatting 
        since this will be read aloud. Aim for responses under 100 words unless 
        specifically asked for detailed information.
        """
        )
        self.conversation_history = [system_msg]
        print("✅ Claude initialized")

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
        """Transcribe audio using Whisper."""
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

                # Transcribe with explicit FP32 to avoid warning
                result = self.whisper_model.transcribe(
                    tmp_file.name,
                    language="en",
                    fp16=False,  # Explicitly use FP32 to avoid CPU warning
                )
                text = result["text"].strip()

                # Clean up
                os.unlink(tmp_file.name)

                return text

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

    async def get_claude_response(self, user_text: str) -> str:
        """Get response from Claude."""
        try:
            # Add user message to history
            self.conversation_history.append(HumanMessage(content=user_text))

            # Get Claude's response
            response = await self.claude.ainvoke(self.conversation_history)
            response_text = response.content

            # Add Claude's response to history
            self.conversation_history.append(AIMessage(content=response_text))

            # Keep conversation history manageable (last 10 messages + system)
            if len(self.conversation_history) > 11:
                self.conversation_history = [
                    self.conversation_history[0]
                ] + self.conversation_history[-10:]

            return response_text

        except Exception as e:
            print(f"❌ Claude API error: {e}")
            return "Sorry, I'm having trouble processing that right now."

    def synthesize_speech(self, text: str) -> bool:
        """Convert text to speech using Piper TTS."""
        try:
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

            return True

        except Exception as e:
            print(f"❌ Speech synthesis error: {e}")
            import traceback

            traceback.print_exc()
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

                    # Detect speech
                    has_speech = self.detect_speech(vad_chunk)

                    if has_speech:
                        if not self.is_recording:
                            print("🗣️  Speech detected, recording...")
                            self.is_recording = True
                            self.audio_buffer = []

                        # Add the original audio chunk to buffer
                        self.audio_buffer.extend(audio_chunk)
                        self.silence_counter = 0

                    else:
                        if self.is_recording:
                            self.silence_counter += 1
                            self.audio_buffer.extend(audio_chunk)

                            # Check if we've had enough silence to stop recording
                            silence_duration = (
                                self.silence_counter * self.chunk_duration
                            )
                            if silence_duration >= self.silence_duration:
                                print("🤫 Silence detected, processing speech...")
                                await self._process_speech()
                                self.is_recording = False
                                self.silence_counter = 0

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

            # Transcribe
            print("🎯 Transcribing...")
            text = self.transcribe_audio(audio_array)

            if text:
                print(f"📝 Transcription: {text}")
                print(f"👤 You said: {text}")

                # Queue for Claude processing
                await self.response_queue.put(text)
            else:
                print("❓ Could not understand speech")

        except Exception as e:
            print(f"❌ Speech processing error: {e}")

    async def handle_responses(self):
        """Handle Claude responses and TTS."""
        while True:
            try:
                # Wait for user text from speech processing
                user_text = await self.response_queue.get()

                print("🤖 Getting Claude response...")
                response = await self.get_claude_response(user_text)
                print(f"🤖 Claude: {response}")

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
    # Configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return

    # Paths to Piper model files (download from https://github.com/rhasspy/piper/releases)
    PIPER_MODEL_PATH = "./piper_models/en_US-hfc_female-medium.onnx"
    PIPER_CONFIG_PATH = "./piper_models/en_US-hfc_female-medium.onnx.json"

    if not os.path.exists(PIPER_MODEL_PATH):
        print(f"❌ Piper model not found at {PIPER_MODEL_PATH}")
        print("Download models from: https://github.com/rhasspy/piper/releases")
        return

    if not os.path.exists(PIPER_CONFIG_PATH):
        print(f"❌ Piper config not found at {PIPER_CONFIG_PATH}")
        print("Make sure to download both .onnx and .json files")
        return

    # Create and start the voice chat agent
    agent = VoiceChatAgent(
        anthropic_api_key=ANTHROPIC_API_KEY,
        whisper_model="base",  # or "tiny" for faster processing
        piper_model_path=PIPER_MODEL_PATH,
        piper_config_path=PIPER_CONFIG_PATH,
        speech_threshold=0.5,  # Adjust based on your environment
        silence_duration=1.0,  # Seconds of silence before processing
    )

    await agent.start_listening()


if __name__ == "__main__":
    asyncio.run(main())
