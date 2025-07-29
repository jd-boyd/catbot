#!/usr/bin/env python3
"""
Voice Chat Agent using Silero VAD, Whisper, Claude Sonnet 4, and Piper TTS
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
import re
import platform
import subprocess
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import whisper
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import strip_markdown

# Conditionally import Piper only on non-Darwin platforms
if platform.system() != "Darwin":
    from piper import PiperVoice


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
        piper_model_path: Optional[str] = "./piper_models/en_US-hfc_female-medium.onnx",
        piper_config_path: Optional[str] = "./piper_models/en_US-hfc_female-medium.onnx.json",
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        speech_threshold: float = 0.5,
        silence_duration: float = 1.0,
        playback_audio: bool = False,
        input_device: Optional[int] = None,
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
            playback_audio: Whether to play back recorded audio before transcription
            input_device: Audio input device index to use (None for system default)
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
        self.user_input_device = input_device
        
        # Set defaults for Piper paths on Darwin if not provided
        if platform.system() == "Darwin":
            # On macOS, Piper is optional - set to None if default paths don't exist
            if (self.piper_model_path == "./piper_models/en_US-hfc_female-medium.onnx" and 
                not os.path.exists(self.piper_model_path)):
                self.piper_model_path = None
            if (self.piper_config_path == "./piper_models/en_US-hfc_female-medium.onnx.json" and 
                not os.path.exists(self.piper_config_path)):
                self.piper_config_path = None

        # Initialize components
        self._init_vad()
        self._init_whisper(whisper_model)
        self._init_claude(anthropic_api_key)
        self._init_tts()
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
        if platform.system() == "Darwin":
            print("Using Apple TTS (macOS system TTS)")
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
        You are a helpful voice assistant. Keep your responses very concise and 
        conversational for speech. Avoid lists or complex formatting since this 
        will be read aloud. Aim for responses under 50 words unless specifically 
        asked for detailed information. Be direct and to the point.
        """
        )
        self.conversation_history = [system_msg]
        print("✅ Claude initialized")

    def _init_tts(self):
        """Initialize TTS system based on platform."""
        if platform.system() == "Darwin":
            print("Using Apple TTS (macOS system TTS)...")
            self.tts_system = "apple"
            print("✅ Apple TTS ready")
        else:
            print("Loading Piper TTS model...")
            try:
                self.piper_voice = PiperVoice.load(
                    self.piper_model_path,
                    config_path=self.piper_config_path,
                    use_cuda=torch.cuda.is_available(),
                )
                self.tts_system = "piper"
                print("✅ Piper TTS loaded")
            except Exception as e:
                print(f"❌ Failed to load Piper TTS: {e}")
                print("Make sure you have both the .onnx model file and .json config file")
                raise

    def _init_audio(self):
        """Initialize PyAudio for microphone input."""
        try:
            self.p = pyaudio.PyAudio()
            
            # Store default input device info for later use
            self.default_input_device_index = None
            
            # Check if we're on macOS and show helpful info
            if platform.system() == "Darwin":
                print("🍎 Detected macOS - checking audio permissions...")
                # Try to get default input device to test permissions
                try:
                    default_input = self.p.get_default_input_device_info()
                    self.default_input_device_index = default_input['index']
                    print(f"✅ System default input device: {default_input['name']} (index: {default_input['index']})")
                except Exception as e:
                    print(f"⚠️  Audio permission issue on macOS: {e}")
                    print("💡 Grant microphone permissions in System Preferences > Security & Privacy > Privacy > Microphone")
            else:
                # For non-macOS systems, also try to get default input
                try:
                    default_input = self.p.get_default_input_device_info()
                    self.default_input_device_index = default_input['index']
                    print(f"✅ System default input device: {default_input['name']} (index: {default_input['index']})")
                except Exception as e:
                    print(f"⚠️  Could not get default input device: {e}")
            
            print("✅ Audio system initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize audio system: {e}")
            if platform.system() == "Darwin":
                print("💡 macOS troubleshooting:")
                print("   1. Install PortAudio: brew install portaudio")
                print("   2. Reinstall PyAudio: pip uninstall pyaudio && pip install pyaudio")
                print("   3. Grant microphone permissions in System Preferences")
            raise

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
        """Convert text to speech using platform-specific TTS."""
        try:
            # Clean text to remove markdown and prevent SSML conflicts
            clean_text = self.clean_text_for_tts(text)

            if not clean_text.strip():
                print("⚠️  No text to synthesize after cleaning")
                return False

            print(f"🧹 Cleaned text: {clean_text}")

            if self.tts_system == "apple":
                return self._synthesize_with_apple_tts(clean_text)
            else:
                return self._synthesize_with_piper(clean_text)

        except Exception as e:
            print(f"❌ Speech synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _synthesize_with_apple_tts(self, text: str) -> bool:
        """Synthesize speech using Apple's system TTS (say command)."""
        try:
            # Use macOS 'say' command for TTS
            # The say command speaks the text directly through system audio
            process = subprocess.run([
                "say", 
                text
            ], check=True, capture_output=True, text=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Apple TTS error: {e}")
            return False
        except FileNotFoundError:
            print("❌ Apple TTS not available - 'say' command not found")
            return False

    def _synthesize_with_piper(self, text: str) -> bool:
        """Synthesize speech using Piper TTS."""
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
            print(f"❌ Piper TTS error: {e}")
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

                # macOS-specific audio output configuration
                output_device_index = None
                if platform.system() == "Darwin":
                    # On macOS, explicitly find a working output device
                    for i in range(self.p.get_device_count()):
                        dev_info = self.p.get_device_info_by_index(i)
                        if dev_info['maxOutputChannels'] > 0:
                            output_device_index = i
                            break

                # Open audio stream
                stream = self.p.open(
                    format=self.p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                    output_device_index=output_device_index,
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
            if platform.system() == "Darwin":
                print("💡 If audio playback fails on macOS, try adjusting System Preferences > Sound > Output")

    async def audio_stream_generator(self) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio chunks from microphone asynchronously."""
        try:
            # Prioritize user-specified device, then system default, then fallback
            input_device_index = None
            
            if self.user_input_device is not None:
                # Use user-specified device
                try:
                    dev_info = self.p.get_device_info_by_index(self.user_input_device)
                    if dev_info['maxInputChannels'] > 0:
                        input_device_index = self.user_input_device
                        print(f"🎤 Using user-specified input device: {dev_info['name']} (index: {input_device_index})")
                    else:
                        print(f"⚠️  User-specified device {self.user_input_device} has no input channels, falling back...")
                except Exception as e:
                    print(f"⚠️  User-specified device {self.user_input_device} not available: {e}")
            
            if input_device_index is None:
                # Use system default input device if available
                input_device_index = self.default_input_device_index
                
                if input_device_index is not None:
                    # Confirm we're using the system default
                    dev_info = self.p.get_device_info_by_index(input_device_index)
                    print(f"🎤 Using system default input device: {dev_info['name']} (index: {input_device_index})")
                else:
                    # Fallback: find the first working input device
                    print("⚠️  System default input device not available, searching for alternatives...")
                    for i in range(self.p.get_device_count()):
                        dev_info = self.p.get_device_info_by_index(i)
                        if dev_info['maxInputChannels'] > 0:
                            input_device_index = i
                            print(f"🎤 Fallback to input device: {dev_info['name']} (index: {i})")
                            break
            
            stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
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
                
        except Exception as e:
            print(f"❌ Failed to create audio stream: {e}")
            if platform.system() == "Darwin":
                print("💡 macOS audio troubleshooting:")
                print("   1. Check microphone permissions in System Preferences")
                print("   2. Try running: sudo xcode-select --install")
                print("   3. Restart the application after granting permissions")
            raise

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
                            vad_chunk_duration = self.vad_chunk_size / self.sample_rate
                            silence_duration = self.silence_counter * vad_chunk_duration
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
            
            # Clear the audio buffer to prevent interference with next recording
            self.audio_buffer = []

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

                # Queue for Claude processing
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


def list_audio_devices():
    """List all available audio input and output devices."""
    try:
        p = pyaudio.PyAudio()
        
        print("📻 Available Audio Devices:")
        print("=" * 50)
        
        default_input = None
        default_output = None
        
        try:
            default_input = p.get_default_input_device_info()
        except:
            pass
            
        try:
            default_output = p.get_default_output_device_info()
        except:
            pass
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            device_type = []
            
            if info['maxInputChannels'] > 0:
                device_type.append("INPUT")
            if info['maxOutputChannels'] > 0:
                device_type.append("OUTPUT")
                
            type_str = "/".join(device_type) if device_type else "NONE"
            
            default_markers = []
            if default_input and info['index'] == default_input['index']:
                default_markers.append("DEFAULT INPUT")
            if default_output and info['index'] == default_output['index']:
                default_markers.append("DEFAULT OUTPUT")
                
            default_str = f" ({', '.join(default_markers)})" if default_markers else ""
            
            print(f"  {i:2d}: {info['name']:<30} [{type_str}]{default_str}")
            
        p.terminate()
        
    except Exception as e:
        print(f"❌ Error listing audio devices: {e}")


def check_platform_requirements():
    """Check platform-specific requirements and provide helpful guidance."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print("🍎 Detected macOS")
        
        # Check if PortAudio is likely installed (basic check)
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            p.terminate()
            print(f"✅ Audio system OK - {device_count} audio devices detected")
        except Exception as e:
            print(f"⚠️  Audio system issue: {e}")
            print("💡 Try: brew install portaudio && pip uninstall pyaudio && pip install pyaudio")
            
    elif system == "Linux":
        print("🐧 Detected Linux")
        print("💡 If you encounter audio issues, you may need: sudo apt-get install portaudio19-dev")
    else:
        print(f"💻 Detected {system} - basic compatibility expected")


async def main():
    """Main function to run the voice chat agent."""
    # Check platform requirements
    check_platform_requirements()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Voice Chat Agent with Claude")
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
        "--input-device",
        type=int,
        help="Audio input device index (use --list-devices to see available devices)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )

    args = parser.parse_args()

    # Handle list devices request
    if args.list_devices:
        list_audio_devices()
        return

    # Configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return

    # Paths to Piper model files (download from https://github.com/rhasspy/piper/releases)
    PIPER_MODEL_PATH = "./piper_models/en_US-hfc_female-medium.onnx"
    PIPER_CONFIG_PATH = "./piper_models/en_US-hfc_female-medium.onnx.json"

    # On Darwin, Piper is optional since we can use Apple TTS
    if platform.system() != "Darwin":
        if not os.path.exists(PIPER_MODEL_PATH):
            print(f"❌ Piper model not found at {PIPER_MODEL_PATH}")
            print("Download models from: https://github.com/rhasspy/piper/releases")
            return

        if not os.path.exists(PIPER_CONFIG_PATH):
            print(f"❌ Piper config not found at {PIPER_CONFIG_PATH}")
            print("Make sure to download both .onnx and .json files")
            return
    else:
        # On macOS, check if Piper models exist but don't require them
        if not os.path.exists(PIPER_MODEL_PATH) or not os.path.exists(PIPER_CONFIG_PATH):
            print("ℹ️  Piper models not found - using Apple TTS instead")
            PIPER_MODEL_PATH = None
            PIPER_CONFIG_PATH = None

    # Show configuration
    if args.playback_audio:
        print(
            "🔊 Audio playback enabled - recorded audio will be played back before transcription"
        )

    # Create and start the voice chat agent
    agent = VoiceChatAgent(
        anthropic_api_key=ANTHROPIC_API_KEY,
        whisper_model=args.whisper_model,
        piper_model_path=PIPER_MODEL_PATH,
        piper_config_path=PIPER_CONFIG_PATH,
        speech_threshold=args.speech_threshold,
        silence_duration=args.silence_duration,
        playback_audio=args.playback_audio,
        input_device=args.input_device,
    )

    await agent.start_listening()


if __name__ == "__main__":
    asyncio.run(main())
