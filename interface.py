import tkinter as tk
from tkVideoPlayer import TkinterVideo

class VideoPairUI:
    def __init__(self, pairs):
        self.pairs = pairs
        self.results = []
        self.current_pair = 0

        self.root = tk.Tk()
        self.root.title("Video Comparison")

        self.left_canvas = tk.Canvas(self.root)
        self.right_canvas = tk.Canvas(self.root)

        self.left_canvas.grid(row=0, column=0, padx=20, pady=20)
        self.right_canvas.grid(row=0, column=2, padx=20, pady=20)

        # self.left_video = TkinterVideo(master=self.left_canvas)
        # self.right_video = TkinterVideo(master=self.right_canvas)
        #
        # self.left_video.place(width=350, height=250)
        # self.right_video.place(width=350, height=250)

        self.start_button = tk.Button(self.root, text="Start", command=self.start)
        self.left_button = tk.Button(self.root, text="Left better", command=lambda p="left": self.add_preference(p))
        self.right_button = tk.Button(self.root, text="Right better", command=lambda p="right": self.add_preference(p))
        self.equal_button = tk.Button(self.root, text="Equal", command=lambda p="equal": self.add_preference(p))
        self.skip_button = tk.Button(self.root, text="Skip", command=lambda p="skip": self.add_preference(p))

        self.start_button.grid(row=0, column=1, padx=20, pady=20)
        self.left_button.grid(row=1, column=0, padx=20, pady=20)
        self.right_button.grid(row=1, column=2, padx=20, pady=20)
        self.equal_button.grid(row=1, column=1, padx=20, pady=20)
        self.skip_button.grid(row=2, column=1, padx=20, pady=20)

        self.place_videos()

        # self.left_video.bind('<<Ended>>', self.loop_left_video)
        # self.right_video.bind('<<Ended>>', self.loop_right_video)

        self.root.mainloop()

    def place_videos(self):
        self.left_video = TkinterVideo(master=self.left_canvas)
        self.right_video = TkinterVideo(master=self.right_canvas)

        self.left_video.place(width=350, height=250)
        self.right_video.place(width=350, height=250)

    def start(self):
        self.load_pair()
        self.root.grid_slaves(row=0, column=1)[0].destroy()

    def add_preference(self, preference):
        self.results.append(preference)
        self.next_pair()

    # def loop_left_video(self, event):
    #     self.left_video.play()
    #
    # def loop_right_video(self, event):
    #     self.right_video.play()

    def load_pair(self):
        pair = self.pairs[self.current_pair]

        self.left_video.load(pair[0])
        self.left_video.play()

        self.right_video.load(pair[1])
        self.right_video.play()

    def next_pair(self):
        self.current_pair += 1
        if self.current_pair >= len(self.pairs):
            self.root.destroy()
        else:
            self.left_video.destroy()
            self.right_video.destroy()
            self.place_videos()
            # self.left_video.stop()
            # self.right_video.stop()
            self.load_pair()

pairs = [("videos/rl-video-episode-0.mp4", "videos/rl-video-episode-500.mp4"),
         ("videos/rl-video-episode-1000.mp4", "videos/rl-video-episode-1500.mp4"),
         ("videos/rl-video-episode-2000.mp4", "videos/rl-video-episode-2995.mp4")]
interface = VideoPairUI(pairs)
print("User choices:", interface.results)