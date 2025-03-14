import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import argparse
import logging
# Import local functions
from functions import GeneralFunctions


GeneralFunctions().setup_logging()


# Dataset class for training on real songs using provided directories.
class RealVocalSpeechDataset(Dataset):
    def __init__(self, mixture_dir, vocal_dir, sample_rate=16000, duration=5.0):
        """
        mixture_dir: Directory containing mixture audio files.
        vocal_dir: Directory containing isolated vocal audio files.
        sample_rate: Target sample rate for audio loading.
        duration: Duration (in seconds) to which each audio is cropped or padded.
        """
        self.mixture_dir = mixture_dir
        self.vocal_dir = vocal_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        
        # List all .wav files in the mixture directory (assume matching names in vocal_dir)
        self.file_names = sorted([f for f in os.listdir(mixture_dir) if f.lower().endswith('.wav')])
        if len(self.file_names) == 0:
            raise ValueError(f"No .wav files found in {mixture_dir}")
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        mixture_path = os.path.join(self.mixture_dir, file_name)
        vocal_path = os.path.join(self.vocal_dir, file_name)
        
        # Load audio files with librosa
        mixture_audio, _ = librosa.load(mixture_path, sr=self.sample_rate)
        vocal_audio, _ = librosa.load(vocal_path, sr=self.sample_rate)
        
        # Crop or pad audio to a fixed length
        mixture_audio = self._fix_length(mixture_audio)
        vocal_audio = self._fix_length(vocal_audio)
        
        # Compute the complementary source (e.g. speech or accompaniment)
        speech_audio = mixture_audio - vocal_audio
        
        # Convert to torch tensors
        mixture_tensor = torch.tensor(mixture_audio, dtype=torch.float32)
        vocal_tensor = torch.tensor(vocal_audio, dtype=torch.float32)
        speech_tensor = torch.tensor(speech_audio, dtype=torch.float32)
        
        # Target: 2-channel tensor; channel 0 = vocals, channel 1 = speech.
        target = torch.stack([vocal_tensor, speech_tensor], dim=0)
        
        return mixture_tensor, target

    def _fix_length(self, audio):
        if len(audio) > self.num_samples:
            return audio[:self.num_samples]
        elif len(audio) < self.num_samples:
            return np.pad(audio, (0, self.num_samples - len(audio)))
        return audio

# Model with a two-channel decoder output (channel 0: vocals, channel 1: speech)
class VocalSeparatorModel(nn.Module):
    def __init__(self):
        super(VocalSeparatorModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=2, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
        # x: (batch_size, signal_length)
        x = x.unsqueeze(1)  # becomes (batch_size, 1, signal_length)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)  # (batch_size, 2, output_length)
        return decoded

# Trainer class for training the two-channel model.
class VocalSpeechExtractorTrainer:
    def __init__(self, model, train_loader, criterion, optimizer, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for mixture, targets in self.train_loader:
                mixture = mixture.to(self.device)
                targets = targets.to(self.device)  # shape: (batch_size, 2, signal_length)
                
                outputs = self.model(mixture)  # shape: (batch_size, 2, output_length)
                loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            avg_loss = running_loss / len(self.train_loader)
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        logging.info(f"Model saved to {save_path}")

# Extractor class for using the trained model to separate vocals and speech.
class VocalSpeechExtractor:
    def __init__(self, model_path, device='cpu', sample_rate=16000):
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.model = VocalSeparatorModel().to(self.device)
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logging.info(f"Model loaded from {model_path}")

    @staticmethod
    def adjust_length(output, target_length):
        output_length = len(output)
        if output_length > target_length:
            return output[:target_length]
        elif output_length < target_length:
            return np.pad(output, (0, target_length - output_length))
        return output

    def extract_sources(self, mixture_file, output_vocals_file, output_speech_file):
        # Load the entire mixture audio file.
        mixture_audio, sr = librosa.load(mixture_file, sr=self.sample_rate)
        original_length = len(mixture_audio)
        
        mixture_tensor = torch.tensor(mixture_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(mixture_tensor)
        outputs = outputs.squeeze(0)  # shape: (2, output_length)
        
        vocals = outputs[0].cpu().numpy()
        speech = outputs[1].cpu().numpy()
        
        vocals = self.adjust_length(vocals, original_length)
        speech = self.adjust_length(speech, original_length)
        
        sf.write(output_vocals_file, vocals, self.sample_rate)
        sf.write(output_speech_file, speech, self.sample_rate)
        logging.info(f"Extracted vocals saved to {output_vocals_file}")
        logging.info(f"Extracted speech saved to {output_speech_file}")

def main():
    parser = argparse.ArgumentParser(description="Train or Extract using a two-channel VocalSpeech model")
    parser.add_argument('--mode', type=str, choices=['train', 'extract'], required=True,
                        help="Mode: 'train' to retrain the model, 'extract' to separate sources from an audio file")
    parser.add_argument('--model_path', type=str, default='models/vocal_separator_model.pth',
                        help="Path to save/load the model")
    # Arguments for extraction mode
    parser.add_argument('--mixture_file', type=str, default='data/mixtures/song_example.wav',
                        help="Path to the mixture audio file (for extraction)")
    parser.add_argument('--output_vocals_file', type=str, default='extracted_vocals.wav',
                        help="Output file path for vocals (for extraction)")
    parser.add_argument('--output_speech_file', type=str, default='extracted_speech.wav',
                        help="Output file path for speech (for extraction)")
    # Arguments for training mode
    parser.add_argument('--mixture_dir', type=str, default='/mnt/c/Users/tsepe/Music/speech',
                        help="Directory containing mixture audio files (for training)")
    parser.add_argument('--vocal_dir', type=str, default='/mnt/c/Users/tsepe/Music/vocals',
                        help="Directory containing isolated vocal audio files (for training)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for training")
    parser.add_argument('--duration', type=float, default=5.0, help="Duration (in seconds) for training samples")
    parser.add_argument('--sample_rate', type=int, default=16000, help="Sample rate for audio processing")
    args = parser.parse_args()

    if args.mode == 'train':
        # Use real songs from the specified directories.
        dataset = RealVocalSpeechDataset(args.mixture_dir, args.vocal_dir,
                                         sample_rate=args.sample_rate, duration=args.duration)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        model = VocalSeparatorModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        trainer = VocalSpeechExtractorTrainer(model, train_loader, criterion, optimizer, device='cpu')
        trainer.train(num_epochs=args.epochs)
        trainer.save_model(args.model_path)

    elif args.mode == 'extract':
        extractor = VocalSpeechExtractor(args.model_path, device='cpu', sample_rate=args.sample_rate)
        extractor.extract_sources(args.mixture_file, args.output_vocals_file, args.output_speech_file)

if __name__ == '__main__':
    main()
    
    # ******** USAGE ********
    # `python src/ml_vocal_speech_extractor.py --mode train --epochs 5 --model_path models/vocal_separator_model.pth --duration 5.0 --sample_rate 16000`