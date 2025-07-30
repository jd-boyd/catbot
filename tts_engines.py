"""
Text-to-Speech Engine API for Voice Chat Agent
Provides a modular interface for different TTS implementations.
"""

import tempfile
import time
import wave
import platform
import pyaudio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""
    
    def __init__(self, **kwargs):
        """Initialize TTS engine with configuration parameters."""
        self.is_speaking = False
        self.config = kwargs
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the TTS engine and load models.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes (WAV format) or None if failed
        """
        pass
    
    def get_sample_rate(self) -> int:
        """Get the sample rate for generated audio."""
        return 22050  # Default sample rate
    
    def get_channels(self) -> int:
        """Get number of audio channels."""
        return 1  # Mono by default
    
    def get_sample_width(self) -> int:
        """Get sample width in bytes."""
        return 2  # 16-bit by default
    
    def speak(self, text: str, audio_player=None) -> bool:
        """
        High-level interface to synthesize and play speech.
        
        Args:
            text: Text to speak
            audio_player: Optional audio player function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.is_speaking = True
            
            # Synthesize audio
            audio_data = self.synthesize(text)
            if not audio_data:
                return False
            
            # Play audio
            if audio_player:
                success = audio_player(audio_data, self.get_sample_rate(), 
                                     self.get_channels(), self.get_sample_width())
            else:
                success = self._default_audio_player(audio_data)
            
            # Brief delay to let audio echo dissipate
            time.sleep(0.5)
            
            return success
            
        except Exception as e:
            print(f"❌ TTS speak error: {e}")
            return False
        finally:
            self.is_speaking = False
    
    def _default_audio_player(self, audio_data: bytes) -> bool:
        """Default audio player using temporary file."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                
                # Play using PyAudio
                return self._play_wav_file(tmp_file.name)
                
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
            return False
    
    def _play_wav_file(self, file_path: str) -> bool:
        """Play WAV file using PyAudio."""
        try:
            p = pyaudio.PyAudio()
            
            with wave.open(file_path, "rb") as wav_file:
                # Get audio parameters
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                # Open audio stream
                stream = p.open(
                    format=p.get_format_from_width(sample_width),
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
            
            p.terminate()
            return True
            
        except Exception as e:
            print(f"❌ WAV playback error: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        pass


class AppleTTSEngine(TTSEngine):
    """TTS engine using Apple's built-in TTS via pyttsx3."""
    
    def __init__(self, voice_name: str = "samantha", rate: int = 200, volume: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.voice_name = voice_name.lower()
        self.rate = rate
        self.volume = volume
        self.tts_engine = None
    
    def initialize(self) -> bool:
        """Initialize Apple TTS engine."""
        try:
            if platform.system() != "Darwin":
                print("❌ Apple TTS only available on macOS")
                return False
            
            import pyttsx3
            
            print("Initializing Apple TTS with pyttsx3...")
            self.tts_engine = pyttsx3.init("nsss")  # Use NSSpeechSynthesizer
            
            # Set voice
            voices = self.tts_engine.getProperty("voices")
            for voice in voices:
                if self.voice_name in voice.name.lower():
                    self.tts_engine.setProperty("voice", voice.id)
                    break
            
            # Set speech rate and volume
            self.tts_engine.setProperty("rate", self.rate)
            self.tts_engine.setProperty("volume", self.volume)
            
            print(f"✅ Apple TTS initialized with voice: {self.voice_name}")
            return True
            
        except Exception as e:
            print(f"❌ Apple TTS initialization failed: {e}")
            return False
    
    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize speech using Apple TTS (returns None as it plays directly)."""
        try:
            if not self.tts_engine:
                return None
            
            # Apple TTS plays directly, we don't get audio data back
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
            # Return empty bytes to indicate success
            return b""
            
        except Exception as e:
            print(f"❌ Apple TTS synthesis error: {e}")
            return None
    
    def speak(self, text: str, audio_player=None) -> bool:
        """Override speak method for Apple TTS direct playback."""
        try:
            self.is_speaking = True
            
            if not self.tts_engine:
                return False
            
            # Apple TTS handles playback directly
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
            # Brief delay to let audio echo dissipate
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"❌ Apple TTS speak error: {e}")
            return False
        finally:
            self.is_speaking = False


class PiperTTSEngine(TTSEngine):
    """TTS engine using Piper TTS."""
    
    def __init__(self, model_path: str, config_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.config_path = config_path
        self.piper_voice = None
    
    def initialize(self) -> bool:
        """Initialize Piper TTS engine."""
        try:
            from piper import PiperVoice
            import torch
            
            print("Loading Piper TTS model...")
            self.piper_voice = PiperVoice.load(
                self.model_path,
                config_path=self.config_path,
                use_cuda=torch.cuda.is_available(),
            )
            print("✅ Piper TTS loaded")
            return True
            
        except Exception as e:
            print(f"❌ Piper TTS initialization failed: {e}")
            return False
    
    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize speech using Piper TTS."""
        try:
            if not self.piper_voice:
                return None
            
            # Generate audio using Piper
            audio_bytes = b""
            for audio_chunk in self.piper_voice.synthesize(text):
                audio_bytes += audio_chunk.audio_int16_bytes
            
            # Create WAV file data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                with wave.open(tmp_file.name, "wb") as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.piper_voice.config.sample_rate)
                    wav_file.writeframes(audio_bytes)
                
                # Read back the complete WAV file
                with open(tmp_file.name, "rb") as wav_read:
                    wav_data = wav_read.read()
                
                return wav_data
                
        except Exception as e:
            print(f"❌ Piper TTS synthesis error: {e}")
            return None
    
    def get_sample_rate(self) -> int:
        """Get Piper model sample rate."""
        if self.piper_voice:
            return self.piper_voice.config.sample_rate
        return 22050


class KokoroTTSEngine(TTSEngine):
    """TTS engine using Kokoro TTS with MLX."""
    
    def __init__(self, model_id: str = 'prince-canuma/Kokoro-82M', 
                 voice: str = 'af_heart', speed: float = 1.0, 
                 lang_code: str = 'a', **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.model = None
        self.pipeline = None
    
    def initialize(self) -> bool:
        """Initialize Kokoro TTS engine."""
        try:
            from mlx_audio.tts.models.kokoro import KokoroPipeline
            from mlx_audio.tts.utils import load_model
            
            print(f"Loading Kokoro TTS model: {self.model_id}...")
            
            # Load model
            self.model = load_model(self.model_id)
            
            # Create pipeline
            self.pipeline = KokoroPipeline(
                lang_code=self.lang_code, 
                model=self.model, 
                repo_id=self.model_id
            )
            
            print(f"✅ Kokoro TTS loaded with voice: {self.voice}")
            return True
            
        except Exception as e:
            print(f"❌ Kokoro TTS initialization failed: {e}")
            print("Make sure mlx-audio is installed: uv pip install mlx-audio")
            return False
    
    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize speech using Kokoro TTS."""
        try:
            if not self.pipeline:
                return None
            
            # Generate audio
            audio_data = None
            for _, _, audio in self.pipeline(
                text, 
                voice=self.voice, 
                speed=self.speed, 
                split_pattern=r'\n+'
            ):
                # Take the first generated audio chunk
                audio_data = audio[0] if isinstance(audio, (list, tuple)) else audio
                break
            
            if audio_data is None:
                return None
            
            # Convert to WAV format
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Write audio using soundfile
                sf.write(tmp_file.name, audio_data, 24000)
                
                # Read back the complete WAV file
                with open(tmp_file.name, "rb") as wav_read:
                    wav_data = wav_read.read()
                
                return wav_data
                
        except Exception as e:
            print(f"❌ Kokoro TTS synthesis error: {e}")
            return None
    
    def get_sample_rate(self) -> int:
        """Get Kokoro model sample rate."""
        return 24000  # Kokoro uses 24kHz


class F5TTSEngine(TTSEngine):
    """TTS engine using F5-TTS with MLX."""
    
    def __init__(self, model_name: str = "lucasnewman/f5-tts-mlx", 
                 ref_audio_path: Optional[str] = None, 
                 ref_audio_text: Optional[str] = None,
                 speed: float = 1.0, 
                 steps: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.ref_audio_path = ref_audio_path
        self.ref_audio_text = ref_audio_text
        self.speed = speed
        self.steps = steps
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize F5-TTS engine."""
        try:
            from f5_tts_mlx.generate import generate
            
            print(f"Initializing F5-TTS engine: {self.model_name}...")
            
            # Test basic import - F5-TTS models are loaded on-demand
            self.generate_func = generate
            self.is_initialized = True
            
            print("✅ F5-TTS engine initialized")
            return True
            
        except ImportError as e:
            print(f"❌ F5-TTS initialization failed: {e}")
            print("Make sure f5-tts-mlx is installed: uv pip install f5-tts-mlx")
            return False
        except Exception as e:
            print(f"❌ F5-TTS initialization failed: {e}")
            return False
    
    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize speech using F5-TTS."""
        try:
            if not self.is_initialized:
                return None
            
            # Generate audio using F5-TTS with correct parameters
            generate_kwargs = {
                "generation_text": text,  # Correct parameter name
                "model_name": self.model_name,
                "speed": self.speed,
                "steps": self.steps
            }
            
            # Add reference audio if provided
            if self.ref_audio_path:
                generate_kwargs["ref_audio_path"] = self.ref_audio_path
            if self.ref_audio_text:
                generate_kwargs["ref_audio_text"] = self.ref_audio_text
            
            # Generate to temporary file (F5-TTS can write directly to file)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                generate_kwargs["output_path"] = tmp_file.name
                
                # Generate audio
                audio_data = self.generate_func(**generate_kwargs)
                
                # Read back the WAV file
                with open(tmp_file.name, "rb") as wav_read:
                    wav_data = wav_read.read()
                
                # Clean up temp file
                import os
                os.unlink(tmp_file.name)
                
                return wav_data
                
        except Exception as e:
            print(f"❌ F5-TTS synthesis error: {e}")
            return None
    
    def get_sample_rate(self) -> int:
        """Get F5-TTS sample rate."""
        return 24000  # F5-TTS typically uses 24kHz


# TTS Factory
class TTSFactory:
    """Factory class for creating TTS engines."""
    
    @staticmethod
    def create_engine(engine_type: str, **kwargs) -> Optional[TTSEngine]:
        """
        Create TTS engine of specified type.
        
        Args:
            engine_type: Type of engine ('apple', 'piper', 'kokoro')
            **kwargs: Engine-specific configuration parameters
            
        Returns:
            Initialized TTS engine or None if failed
        """
        engines = {
            'apple': AppleTTSEngine,
            'piper': PiperTTSEngine,
            'kokoro': KokoroTTSEngine,
            'f5': F5TTSEngine,
        }
        
        if engine_type not in engines:
            print(f"❌ Unknown TTS engine type: {engine_type}")
            print(f"Available engines: {list(engines.keys())}")
            return None
        
        try:
            engine = engines[engine_type](**kwargs)
            if engine.initialize():
                return engine
            else:
                return None
        except Exception as e:
            print(f"❌ Failed to create {engine_type} TTS engine: {e}")
            return None
    
    @staticmethod
    def get_recommended_engine() -> str:
        """Get recommended TTS engine for current platform."""
        if platform.system() == "Darwin":
            return "apple"
        else:
            return "piper"