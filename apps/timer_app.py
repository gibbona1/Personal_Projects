import tkinter as tk
from threading import Thread, Event
import time
import pygame

class CountdownApp:
    def __init__(self, master):
        self.master = master
        master.title("Countdown Timer")

        self.time_left = 10.000  # 10 seconds countdown with milliseconds
        self.original_time = self.time_left
        self.running_event = Event()

        self.display = tk.Label(master, text="10.000", font=("Helvetica", 48))
        self.display.pack()

        self.start_button = tk.Button(master, text="Start", command=self.start_timer)
        self.start_button.pack()

        self.relay_button = tk.Button(master, text="Relay", command=self.relay)
        self.relay_button.pack()

        self.reset_button = tk.Button(master, text="Reset", command=self.reset)
        self.reset_button.pack()

        pygame.init()


    def countdown(self):
        start_time = time.perf_counter()
        while not self.running_event.is_set():
            elapsed = time.perf_counter() - start_time
            self.time_left = max(self.original_time - elapsed, 0)

            if self.time_left <= 0:
                self.update_display()
                self.alarm()
                break

            if int(elapsed * 100) % 10 == 0:  # Update display every 10 milliseconds
                self.update_display()

            time.sleep(0.001)  # Sleep for 1 millisecond for responsiveness

    def update_display(self):
        timeformat = '{:.3f}'.format(self.time_left)
        self.display.config(text=timeformat)

    def start_timer(self):
        if not self.running_event.is_set():
            self.running_event.clear()
            self.thread = Thread(target=self.countdown)
            self.thread.start()

    def relay(self):
        self.running_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.running_event.clear()
        self.time_left = self.original_time
        self.update_display()
        self.start_timer()

    def reset(self):
        self.running_event.set()
        self.time_left = self.original_time
        self.update_display()

    def alarm(self):
        pygame.mixer.music.load("alarm.mp3")  # Ensure you have an alarm.mp3 file
        pygame.mixer.music.play()
        # Implement the alarm functionality here.
        print("Alarm!")  # Placeholder for alarm functionality

def main():
    root = tk.Tk()
    app = CountdownApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
