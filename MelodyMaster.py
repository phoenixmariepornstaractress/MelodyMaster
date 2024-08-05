import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pygame
import os
import librosa
import numpy as np
import requests
import json
import time

# Enhanced feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    
    features = np.hstack([
        np.mean(mfccs.T, axis=0), 
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0), 
        np.mean(contrast.T, axis=0), 
        np.mean(tonnetz.T, axis=0)
    ])
    return features

# AI model setup (simplified for demonstration)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

features = np.random.rand(1000, 193)
labels = np.random.randint(2, size=1000)

model = Sequential([
    Dense(256, input_shape=(193,), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(features, labels, epochs=50, batch_size=32, validation_split=0.2)

def recommend(song_features, mood=None):
    prediction = model.predict(np.array([song_features]))
    if mood == "Happy":
        return prediction[0][0] > 0.7
    elif mood == "Sad":
        return prediction[0][0] < 0.3
    else:
        return prediction[0][0] > 0.5

class MusicPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Music Player")
        self.root.geometry("800x600")
        
        pygame.mixer.init()
        
        self.playlist = []
        
        self.create_widgets()
        
        self.song_path = None

    def create_widgets(self):
        self.load_button = tk.Button(self.root, text="Load Song", command=self.load_song)
        self.load_button.pack(pady=10)
        
        self.play_button = tk.Button(self.root, text="Play", command=self.play_song)
        self.play_button.pack(pady=10)
        
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_song)
        self.stop_button.pack(pady=10)
        
        self.pause_button = tk.Button(self.root, text="Pause", command=self.pause_song)
        self.pause_button.pack(pady=10)
        
        self.resume_button = tk.Button(self.root, text="Resume", command=self.resume_song)
        self.resume_button.pack(pady=10)
        
        self.next_button = tk.Button(self.root, text="Next", command=self.next_song)
        self.next_button.pack(pady=10)
        
        self.prev_button = tk.Button(self.root, text="Previous", command=self.prev_song)
        self.prev_button.pack(pady=10)
        
        self.song_listbox = tk.Listbox(self.root)
        self.song_listbox.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.volume_slider = tk.Scale(self.root, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL, label="Volume", command=self.set_volume)
        self.volume_slider.set(0.5)
        self.volume_slider.pack(pady=10)
        
        self.mood_label = tk.Label(self.root, text="Select Mood:")
        self.mood_label.pack(pady=10)
        
        self.mood_var = tk.StringVar(value="Neutral")
        self.mood_options = ["Happy", "Sad", "Neutral"]
        self.mood_menu = tk.OptionMenu(self.root, self.mood_var, *self.mood_options)
        self.mood_menu.pack(pady=10)
        
        self.lyrics_text = tk.Text(self.root, height=10, width=80)
        self.lyrics_text.pack(pady=10)
        
        self.album_art_label = tk.Label(self.root)
        self.album_art_label.pack(pady=10)
        
        self.search_label = tk.Label(self.root, text="Search:")
        self.search_label.pack(pady=10)
        
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(self.root, textvariable=self.search_var)
        self.search_entry.pack(pady=10)
        
        self.search_button = tk.Button(self.root, text="Search", command=self.search_song)
        self.search_button.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', length=300)
        self.progress_bar.pack(pady=10)
        
        self.song_duration_label = tk.Label(self.root, text="Duration: 00:00 / 00:00")
        self.song_duration_label.pack(pady=10)
    
    def load_song(self):
        song_paths = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.mp3 *.wav")])
        for path in song_paths:
            self.playlist.append(path)
            self.song_listbox.insert(tk.END, os.path.basename(path))
        if song_paths:
            self.song_path = self.playlist[0]
            self.display_metadata(self.song_path)

    def play_song(self):
        if self.song_path:
            pygame.mixer.music.load(self.song_path)
            pygame.mixer.music.play()
            self.fetch_lyrics(self.song_path)
            self.display_album_art(self.song_path)
            self.update_progress_bar()

    def stop_song(self):
        pygame.mixer.music.stop()

    def pause_song(self):
        pygame.mixer.music.pause()

    def resume_song(self):
        pygame.mixer.music.unpause()

    def next_song(self):
        current_index = self.playlist.index(self.song_path)
        next_index = (current_index + 1) % len(self.playlist)
        self.song_path = self.playlist[next_index]
        self.play_song()

    def prev_song(self):
        current_index = self.playlist.index(self.song_path)
        prev_index = (current_index - 1) % len(self.playlist)
        self.song_path = self.playlist[prev_index]
        self.play_song()

    def set_volume(self, val):
        volume = float(val)
        pygame.mixer.music.set_volume(volume)

    def recommend_song(self):
        if self.song_path:
            features = extract_features(self.song_path)
            mood = self.mood_var.get()
            if recommend(features, mood):
                messagebox.showinfo("Recommendation", f"This song matches your mood ({mood})!")
            else:
                messagebox.showinfo("Recommendation", f"This song does not match your mood ({mood}).")

    def fetch_lyrics(self, song_path):
        song_name = os.path.basename(song_path).split('.')[0]
        api_url = f"https://api.lyrics.ovh/v1/{song_name}"
        response = requests.get(api_url)
        if response.status_code == 200:
            lyrics = response.json().get('lyrics', "Lyrics not found.")
        else:
            lyrics = "Lyrics not found."
        self.lyrics_text.delete(1.0, tk.END)
        self.lyrics_text.insert(tk.END, lyrics)

    def display_album_art(self, song_path):
        song_name = os.path.basename(song_path).split('.')[0]
        api_url = f"https://api.deezer.com/search?q={song_name}"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data['data']:
                album_cover_url = data['data'][0]['album']['cover']
                response = requests.get(album_cover_url, stream=True)
                if response.status_code == 200:
                    img_data = response.raw
                    img = Image.open(img_data)
                    img = img.resize((150, 150), Image.ANTIALIAS)
                    img = ImageTk.PhotoImage(img)
                    self.album_art_label.config(image=img)
                    self.album_art_label.image = img
                else:
                    self.album_art_label.config(image='')
            else:
                self.album_art_label.config(image='')
        else:
            self.album_art_label.config(image='')

    def search_song(self):
        search_query = self.search_var.get().lower()
        matched_songs = [song for song in self.playlist if search_query in os.path.basename(song).lower()]
        self.song_listbox.delete(0, tk.END)
        for song in matched_songs:
            self.song_listbox.insert(tk.END, os.path.basename(song))

    def update_progress_bar(self):
        if pygame.mixer.music.get_busy():
            song_length = pygame.mixer.Sound(self.song_path).get_length()
            current_pos = pygame.mixer.music.get_pos() / 1000
            self.progress_bar['maximum'] = song_length
            self.progress_bar['value'] = current_pos
            minutes, seconds = divmod(current_pos, 60)
            total_minutes, total_seconds = divmod(song_length, 60)
            self.song_duration_label.config(text=f"Duration: {int(minutes):02}:{int(seconds):02} / {int(total_minutes):02}:{int(total_seconds):02}")
            self.root.after(1000, self.update_progress_bar)

if __name__ == "__main__":
    root = tk.Tk()
    player = MusicPlayer(root)
    root.mainloop()
