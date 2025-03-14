import librosa
from soundfile import SoundFile, write
from moviepy import VideoFileClip, CompositeAudioClip
import logging
from numpy import median, minimum, ndarray, pad
import time
from typing import Union, NoReturn, Tuple
from os import makedirs, path, listdir
import torch
import concurrent
# Import local functions
from functions import GeneralFunctions


GeneralFunctions().setup_logging()


class AudioTools(GeneralFunctions):
    def __init__(self) -> None:
        self.configs_dict = self.load_toml("configs/configs.toml")
        
    def extract_audio_from_video(self, video_path: str, export_audio_locally: bool = False, export_audio_path: str = "exports/audio/audio_only.wav") -> tuple[Union[bytes, NoReturn], Union[str, NoReturn]]:
        """
        Extracts audio from a video file and returns it as bytes or writes it to a file.

        Args:
            video_path (str): The path to the video file to be processed.
            export_audio_locally (bool, optional): If True, the audio is written to a file. Defaults to False.
            export_audio_path (str, optional): The path to write the audio file to. Defaults to "exports/audio/audio_only.wav".

        Returns:
            tuple: A tuple of bytes and a string. If export_audio_locally is False, the bytes is the audio data and the string is None. If export_audio_locally is True, the bytes is an empty string and the string is the path to the written audio file.
        """
        video = VideoFileClip(video_path)
        if export_audio_locally:
            try:
                filename = video_path.split('.')[0].split('/')[-1]
            except Exception as e:
                logging.error(e)
                filename = export_audio_path
            _path = f"exports/audio/{filename}.wav"
            video.audio.write_audiofile(_path)
            logging.info("Extracted audio from video.")
            return b'', _path
        else:
            return video.audio, None
        
    def add_audio_from_local_path_to_binary_io(self, audio_path: str) -> tuple[Union[SoundFile, NoReturn], Union[bool, NoReturn]]:
        """
        Reads an audio file from a local path and returns it as a SoundFile object with a boolean indicating if the file was loaded successfully.

        Args:
            audio_path (str): The path to the audio file to be read.

        Returns:
            tuple: A tuple of a SoundFile object and a boolean. If the file was loaded successfully, the boolean is True and the SoundFile object contains the loaded audio data. If the file failed to load, the boolean is False and the SoundFile object is empty.
        """
        try:
            audio_file = SoundFile(audio_path)
            loaded = True
            logging.info('Loaded audio from local path')
            return audio_file, loaded
        except Exception as e:
            logging.error(e)
            audio_file = SoundFile(b'')
            loaded = False
            logging.error('Failed to load audio from local path')
            return audio_file, loaded
    
    def remove_speech_from_video(self) -> None:
        """
        Removes speech from a video.

        This function extracts the audio from the video, applies a non-negative matrix factorization filter to separate the vocal and instrumental components, and then writes the audio without vocals back into the video.

        Args:
            None

        Returns:
            None
        """
        
        # Extract audio
        start_time = time.time()
        video = VideoFileClip(self.video_path)
        audio_only_path = "static/exports/audio_only.wav"
        video.audio.write_audiofile(audio_only_path)
        logging.info("Extracted audio from video.")
        
        # Extract vocals
        y, sr = librosa.load(audio_only_path)
        S_full, phase = librosa.magphase(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full,
                                            aggregate=median,
                                            metric='cosine',
                                            width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = minimum(S_full, S_filter)
        logging.info("Decomposed.")
        # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
        # Note: the margins need not be equal for foreground and background separation
        margin_i, margin_v = 2, 10
        power = 2

        mask_i = librosa.util.softmask(S_filter,
                                    margin_i * (S_full - S_filter),
                                    power=power)

        mask_v = librosa.util.softmask(S_full - S_filter,
                                    margin_v * S_filter,
                                    power=power)

        S_foreground = mask_v * S_full
        S_background = mask_i * S_full
        
        new_y = librosa.istft(S_background*phase)
        audio_no_speech_path = "static/exports/audio_only_without_speech.wav"
        write(audio_no_speech_path, new_y, sr)

        logging.info('Vocals Separated')
        
        # Replace audio with no speech inside
        video_no_audio = video.without_audio()
        audio_no_speech = CompositeAudioClip([audio_no_speech_path])
        video_no_audio.audio = audio_no_speech
        video_export_path = self.configs_dict['GENERAL']['VIDEO_PATH'].rsplit(".", 1)[0] + " NO SPEECH.mp4"
        video_no_audio.write_videofile(video_export_path)
        end_time = time.time()
        time_lapsed = end_time - start_time
        logging.info(f'Time to remove speech from video: {time_lapsed}')
    
    def extract_vocals_and_speech_from_audio(self, audio: Union[SoundFile, str]) -> Tuple[ndarray, ndarray, int]:
        """
        Extracts vocals and speech from an audio file, separating the vocals and background music.

        Args:
            audio (Union[SoundFile, str]): The audio input, which can be a SoundFile object or a file path as a string.

        Returns:
            tuple: A tuple containing the extracted vocals (vocals_ndarray), speech (speech_ndarray), and their sample rates (vocals_sr, speech_sr).

        This function uses librosa to perform a short-time Fourier transform on the audio data, 
        applies a non-negative matrix factorization filter to separate the vocal and instrumental components, 
        and logs various stages of the process for debugging purposes.
        """

        start_time = time.time()
        # Extract vocals
        if isinstance(audio, str):
            try:
                audio_file = SoundFile(audio)
            except Exception as e:
                logging.error(e)
                return
        elif isinstance(audio, SoundFile):
            audio_file = audio
        elif isinstance(audio, bytes):
            audio_file = SoundFile(audio)
        elif isinstance(audio, ndarray):
            audio_file = audio
        y, sr = librosa.load(audio_file)
        S_full, phase = librosa.magphase(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full,
                                            aggregate=median,
                                            metric='cosine',
                                            width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = minimum(S_full, S_filter)
        logging.info("Decomposed.")
        # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
        # Note: the margins need not be equal for foreground and background separation
        margin_i, margin_v = 5, 15
        power = 2

        mask_i = librosa.util.softmask(S_filter,
                                    margin_i * (S_full - S_filter),
                                    power=power)

        mask_v = librosa.util.softmask(S_full - S_filter,
                                    margin_v * S_filter,
                                    power=power)

        S_foreground = mask_v * S_full
        S_background = mask_i * S_full
        
        vocals_ndarray = librosa.istft(S_background*phase)
        speech_ndarray = librosa.istft(S_foreground*phase)

        logging.info('Vocals and Speech Separated')
        
        # Return vocals and speech ndarrays and their sample rates
        end_time = time.time()
        time_lapsed = end_time - start_time
        logging.info(f'Time to extract vocals and speech from audio: {time_lapsed}')
        return vocals_ndarray, speech_ndarray, sr
    
    def write_audio_to_file(self, audio_data: dict, sample_rate: int, directory: str) -> None:
        """
        Writes audio data to .wav files in specified subdirectories for speech and vocals.

        Args:
            audio_data (dict): A dictionary with keys as filenames and values as ndarrays of audio data.
            directory (str): The base directory to write the audio files to.
        """
        speech_dir = f"{directory}/speech"
        vocals_dir = f"{directory}/vocals"
        makedirs(speech_dir, exist_ok=True)
        makedirs(vocals_dir, exist_ok=True)
        
        for filename, audio_ndarray in audio_data.items():
            if "speech" in filename:
                write(f"{speech_dir}/{filename}.wav", audio_ndarray, sample_rate, subtype='PCM_24')
            elif "vocal" in filename:
                write(f"{vocals_dir}/{filename}.wav", audio_ndarray, sample_rate, subtype='PCM_24')

    def process_file(self, file_name: str, audio_dir: str, extract_audio_from_video: bool = False) -> None:
        file_path = path.join(audio_dir, file_name)
        
        # Extract audio into binary_io
        if extract_audio_from_video:
            audio_bytes, audio_path = self.extract_audio_from_video(file_path)
            if audio_bytes == b'':
                audio_bytes, loaded = self.add_audio_from_local_path_to_binary_io(audio_path)
        else:
            audio_bytes, loaded = self.add_audio_from_local_path_to_binary_io(file_path)
            
        if not loaded:
            return
        
        vocals_ndarray, speech_ndarray, sr = self.extract_vocals_and_speech_from_audio(audio=audio_bytes)
        audio_data = {
            f"{file_name}_speech_only": speech_ndarray,
            f"{file_name}_vocals_only": vocals_ndarray,
        }
        self.write_audio_to_file(audio_data=audio_data, sample_rate=sr, directory="exports/audio")
    
    def main(self, audio_dir: str, extract_audio_from_video: bool = False) -> None:
        """
        Main function to remove speech from audio files in a directory using multithreading.

        Args:
            audio_dir (str): The directory containing audio files to be processed.
            extract_audio_from_video (bool, optional): If True, the audio is extracted from videos. Defaults to False.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda file_name: self.process_file(file_name, audio_dir, extract_audio_from_video), 
                         [file for file in listdir(audio_dir) if file.split('.')[-1] in ['wav', 'mp3', 'ogg', 'flac']])


# Define the updated model architecture to output two channels: vocals and speech.
class VocalSeparatorModel(torch.nn.Module):
    def __init__(self):
        super(VocalSeparatorModel, self).__init__()
        # Encoder: compress the input audio.
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU()
        )
        # Decoder: reconstruct two outputs: vocals (channel 0) and speech (channel 1).
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(in_channels=16, out_channels=2, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
        # x: shape (batch_size, signal_length)
        x = x.unsqueeze(1)  # (batch_size, 1, signal_length)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)  # (batch_size, 2, signal_length_estimated)
        return decoded

# The extractor class now extracts both vocals and speech.
class VocalSpeechExtractor:
    def __init__(self, model_path, device='cpu', sample_rate=16000):
        """
        Initializes the extractor with a pre-trained model.
        :param model_path: Path to the saved model (.pth file).
        :param device: Device to run inference ('cpu' or 'cuda').
        :param sample_rate: Sample rate for audio processing.
        """
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.model = VocalSeparatorModel().to(self.device)
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load the saved model state dictionary."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logging.info("Model loaded from", model_path)

    @staticmethod
    def adjust_length(output, target_length):
        """
        Adjusts the output numpy array to match the target length by cropping or padding.
        :param output: Numpy array from model output.
        :param target_length: Desired output length.
        :return: Adjusted numpy array.
        """
        output_length = len(output)
        if output_length > target_length:
            return output[:target_length]
        elif output_length < target_length:
            return pad(output, (0, target_length - output_length))
        return output

    def extract_sources(self, mixture_file, output_vocals_file, output_speech_file):
        """
        Loads a mixture audio file (any duration), extracts vocals and speech using the model,
        and saves the two outputs separately.
        :param mixture_file: Path to the mixture audio file.
        :param output_vocals_file: Path to save the extracted vocals.
        :param output_speech_file: Path to save the extracted speech.
        """
        # Load the entire audio file (variable duration)
        mixture_audio, sr = librosa.load(mixture_file, sr=self.sample_rate)
        original_length = len(mixture_audio)
        
        # Convert the mixture to a tensor and add a batch dimension
        mixture_tensor = torch.tensor(mixture_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Run inference; the output shape is (batch_size, 2, estimated_length)
        with torch.no_grad():
            outputs = self.model(mixture_tensor)
        outputs = outputs.squeeze(0)  # shape: (2, estimated_length)
        
        # Split the two channels: channel 0 for vocals, channel 1 for speech
        vocals = outputs[0].cpu().numpy()
        speech = outputs[1].cpu().numpy()
        
        # Adjust lengths to match the original input length
        vocals = self.adjust_length(vocals, original_length)
        speech = self.adjust_length(speech, original_length)
        
        # Save the outputs
        write(output_vocals_file, vocals, self.sample_rate)
        write(output_speech_file, speech, self.sample_rate)
        logging.info(f'Extracted vocals saved to {output_vocals_file}')
        logging.info(f'Extracted speech saved to {output_speech_file}')

    
if __name__ == "__main__":
    use_my_model = True
    if not use_my_model:
        audio_tools = AudioTools()
        audio_tools.main(audio_dir='/mnt/c/Users/tsepe/Downloads/BiglyBT/Madonna')
    else:
        # Update these paths with the actual locations of your model and audio file.
        model_path = 'models/vocal_separator_model.pth'
        mixture_file = '/mnt/c/Users/tsepe/Downloads/Madonna_-_La_Isla_Bonita.mp3'
        output_vocals_file = f'exports/audio/vocals/{path.basename(mixture_file)}_extracted_vocals.wav'
        output_speech_file = f'exports/audio/speech/{path.basename(mixture_file)}_extracted_speech.wav'
        
        extractor = VocalSpeechExtractor(model_path, device='cpu', sample_rate=16000)
        extractor.extract_sources(mixture_file, output_vocals_file, output_speech_file)