import tkinter as tk
from tkVideoPlayer import TkinterVideo

class VideoPairUI:
    def __init__(self, pairs):
        self.pairs = pairs
        self.results = []
        self.current_pair = 0

        self.root = tk.Tk()
        self.root.title("Video Comparison")

        self.left = tk.Canvas(self.root)
        self.right = tk.Canvas(self.root)

        self.left.grid(row=0, column=0, padx=20, pady=20)
        self.right.grid(row=0, column=1, padx=20, pady=20)

        self.left_video = TkinterVideo(master=self.left)
        self.right_video = TkinterVideo(master=self.right)


        self.left_video.place(width=350, height=250)
        self.right_video.place(width=350, height=250)


        self.left_button = tk.Button(self.root, text="Left better", command=self.left)
        self.right_button = tk.Button(self.root, text="Right better", command=self.right)
        self.equal_button = tk.Button(self.root, text="Equal", command=self.equal)
        self.skip_button = tk.Button(self.root, text="Skip", command=self.skip)

        self.left_button.grid(row=1, column=0, padx=20, pady=20)
        self.right_button.grid(row=1, column=1, padx=20, pady=20)
        self.equal_button.grid(row=2, column=0, padx=20, pady=20)
        self.skip_button.grid(row=2, column=1, padx=20, pady=20)


        self.left_video.grid_columnconfigure(0, minsize=500)
        self.left_video.grid_rowconfigure(0, minsize=500)

        self.load_pair()

        self.root.mainloop()

    def load_pair(self):
        pair = self.pairs[self.current_pair]

        self.left_video.load(pair[0])
        self.left_video.play()

        self.right_video.load(pair[1])
        self.right_video.play()

    def left(self):
        self.results.append("left")
        self.next_pair()

    def equal(self):
        self.results.append("equal")
        self.next_pair()

    def right(self):
        self.results.append("right")
        self.next_pair()

    def skip(self):
        self.results.append("skip")
        self.next_pair()

    def next_pair(self):
        self.current_pair += 1
        if self.current_pair >= len(self.pairs):
            self.root.destroy()
        else:
            # self.left_video.stop()
            # self.right_video.stop()
            self.load_pair()

pairs = [("videos/rl-video-episode-0.mp4", "videos/rl-video-episode-500.mp4"),
         ("videos/rl-video-episode-500.mp4", "videos/rl-video-episode-1000.mp4"),
         ("videos/rl-video-episode-1500.mp4", "videos/rl-video-episode-2000.mp4")]
interface = VideoPairUI(pairs)
print("User choices:", interface.results)