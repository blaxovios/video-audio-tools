import librosa
import soundfile as sf
import moviepy.editor as mp
import logging
import numpy as np
import time

from src.functions import GeneralFunctions


class AudioTools(GeneralFunctions):
    def __init__(self) -> None:
        self.configs_dict = self.load_toml("configs/configs.toml")
    
    def remove_speech_from_audio(self) -> None:
        # Extract audio
        start_time = time.time()
        video = mp.VideoFileClip(self.video_path)
        audio_only_path = "static/exports/audio_only.wav"
        video.audio.write_audiofile(audio_only_path)
        logging.info("Extracted audio from video.")
        
        # Extract vocals
        y, sr = librosa.load(audio_only_path)
        S_full, phase = librosa.magphase(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full,
                                            aggregate=np.median,
                                            metric='cosine',
                                            width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = np.minimum(S_full, S_filter)
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
        sf.write(audio_no_speech_path, new_y, sr)

        logging.info('Vocals Separated')
        
        # Replace audio with no speech inside
        video_no_audio = video.without_audio()
        audio_no_speech = mp.CompositeAudioClip([audio_no_speech_path])
        video_no_audio.audio = audio_no_speech
        video_export_path = self.configs_dict['GENERAL']['VIDEO_PATH'].rsplit(".", 1)[0] + " NO SPEECH.mp4"
        video_no_audio.write_videofile(video_export_path)
        end_time = time.time()
        time_lapsed = end_time - start_time
        logging.info(f'Time to read multiple files in cloud: {time_lapsed}')
        return
    
    
if __name__ == "__main__":
    audio_tools = AudioTools()
    audio_tools.remove_speech_from_audio()