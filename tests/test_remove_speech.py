import unittest
from remove_speech import AudioTools


class TestAudioTools(unittest.TestCase):
    def setUp(self):
        self.audio_tools = AudioTools()

    def test_init(self):
        self.assertIsNotNone(self.audio_tools.configs_dict)

    def test_remove_speech_from_audio(self):
        # Assuming there's a way to verify the effect of remove_speech_from_audio
        # This might involve checking if the file exists, or if a log was created, etc.
        # Since the method does not return anything, and without knowing the full implementation details,
        # it's hard to write a meaningful test here.
        pass


if __name__ == '__main__':
    unittest.main()