import librosa
from soundfile import SoundFile, write
from moviepy import VideoFileClip, CompositeAudioClip
import logging
from numpy import median, minimum, ndarray
import time
from typing import Union, NoReturn, Tuple
from os import makedirs
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
    
    def write_audio_to_file(self, audio_ndarray: ndarray, sample_rate: int, directory: str, filename: str) -> None:
        """
        Writes a list of audio bytes to .wav files in a specified directory.

        Args:
            audio_bytes_list (List[bytes]): A list of bytes representing audio data.
            directory (str): The directory to write the audio files to.
        """
        makedirs(directory, exist_ok=True)
        write(f"{directory}/{filename}.wav", audio_ndarray, sample_rate, subtype='PCM_24')
                
    def main(self, audio_path: str, extract_audio_from_video: bool = False) -> None:
        """
        Main function to remove speech from audio.

        Args:
            audio_path (str): The path to the audio file to be processed.
            extract_audio_from_video (bool, optional): If True, the audio is extracted from a video. Defaults to False.
        """
        
        # Extract audio into binary_io
        if extract_audio_from_video:
            audio_bytes, audio_path = audio_tools.extract_audio_from_video()
            if audio_bytes == b'':
                audio_bytes, loaded = audio_tools.add_audio_from_local_path_to_binary_io(audio_path)
        else:
            audio_bytes, loaded = audio_tools.add_audio_from_local_path_to_binary_io(audio_path)
            
        if not loaded:
            return
        
        vocals_ndarray, speech_ndarray, sr = audio_tools.extract_vocals_and_speech_from_audio(audio=audio_bytes)
        audio_tools.write_audio_to_file(audio_ndarray=speech_ndarray, sample_rate=sr, directory="exports/audio", filename="speech_only")
        audio_tools.write_audio_to_file(audio_ndarray=vocals_ndarray, sample_rate=sr, directory="exports/audio", filename="vocals_only")
    
    
if __name__ == "__main__":
    audio_tools = AudioTools()
    audio_tools.main(audio_path='/mnt/c/Users/tsepe/Downloads/Madonna_-_La_Isla_Bonita.mp3')