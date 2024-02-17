import sys
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
import cv2
import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
import os
import threading
import pygame
import time

class VoiceThread(QThread):
    command_received = pyqtSignal(str)
    listening_started = pyqtSignal()
    listening_stopped = pyqtSignal()

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.pepper_command_heard = False

    def run(self):
        pygame.mixer.init()  # Initialize pygame mixer for playing sounds
        self.bleep_start = pygame.mixer.Sound("bleep_start.wav")  # Replace with your sound file
        self.bleep_stop = pygame.mixer.Sound("bleep_stop.wav")  # Replace with your sound file

        print("Listening for commands...")
        with sr.Microphone() as source:
            while True:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    command = self.recognizer.recognize_google(audio)
                    print("You said:", command)
                    self.command_received.emit(command)

                    if self.wake_word.lower() in command.lower():
                        self.pepper_command_heard = True
                        self.bleep_start.play()
                        self.listening_started.emit()

                    time.sleep(5)  # Simulate a 5-second timeout

                    if not self.pepper_command_heard:
                        self.bleep_stop.play()
                        self.listening_stopped.emit()
                        print("Sorry, I couldn't understand.")

                    self.pepper_command_heard = False  # Reset for the next iteration

                except sr.UnknownValueError:
                    print("Could not understand audio")

                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")

                except Exception as e:
                    print(f"Error during continuous listening: {e}")

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the camera
        self.camera_index = 0
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # GUI component
        self.live_feed_label = QLabel(self)
        self.live_feed_label.setFixedSize(640, 480)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.live_feed_label)
        self.setLayout(main_layout)

        # Timer for updating the camera feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Voice command setup
        self.recognizer = sr.Recognizer()
        self.gpt_neo_model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
        self.image_caption_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        self.wake_word = "pepper"

        # Create and start the voice thread
        self.voice_thread = VoiceThread(self.recognizer)
        self.voice_thread.command_received.connect(self.voice_command)
        self.voice_thread.listening_started.connect(self.listening_started)
        self.voice_thread.listening_stopped.connect(self.listening_stopped)
        self.voice_thread.start()

    def listening_started(self):
        print("Listening started.")
        # You can add any actions or UI updates here.

    def listening_stopped(self):
        print("Listening stopped.")
        # You can add any actions or UI updates here.

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            bytes_per_line = 3 * 640
            q_image = QImage(frame.data, 640, 480, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.live_feed_label.setPixmap(QPixmap.fromImage(q_image))

    def voice_command(self, command):
        print(f"Command: {command}")
        if self.wake_word.lower() in command.lower():
            self.start_listening()
        elif "What am I looking at" in command:
            self.process_image()
        else:
            description = self.generate_description(command)
            if description.lower().strip() == "could not understand":
                tts = gTTS("Sorry, I couldn't understand.", lang='en')
                tts.save("output.mp3")
                os.system("mpg123 output.mp3")
            else:
                tts = gTTS(description, lang='en')
                tts.save("output.mp3")
                os.system("mpg123 output.mp3")

    def generate_description(self, command):
        snapshot_path = 'snapshot.jpg'
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.imwrite(snapshot_path, frame)

        try:
            generated_text = self.gpt_neo_model(command, max_length=40, num_return_sequences=1, temperature=0.7)[0]['generated_text']
            print("AI Response:", generated_text)
            return generated_text
        except Exception as e:
            print(f"Error generating description: {e}")
            return "Error generating description"

    def process_image(self):
        snapshot_path = 'snapshot.jpg'
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.imwrite(snapshot_path, frame)

        try:
            result = self.image_caption_model(snapshot_path, max_new_tokens=100)
            description = result[0]['generated_text']
            print("Image Caption:", description)
            tts = gTTS(description, lang='en')
            tts.save("output.mp3")
            os.system("mpg123 output.mp3")
        except Exception as e:
            print(f"Error processing image: {e}")

    def start_listening(self):
        pass  # Since we are handling the sound in the VoiceThread

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.setWindowTitle('Camera App')
    window.setGeometry(100, 100, 800, 600)
    window.show()

    sys.exit(app.exec_())
