"""用鼠标绘制数字，并调用 mnist_classifier.py 训练出的模型识别。"""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from mnist_classifier import DEFAULT_OUTPUT_DIR, MEAN, STD, SimpleMLP


CANVAS_SIZE = 560
LINE_WIDTH = max(8, CANVAS_SIZE // 14)


class DigitCanvas:
    def __init__(self, root: tk.Tk, checkpoint: Path) -> None:
        self.model = SimpleMLP()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.eval()
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.drawer = ImageDraw.Draw(self.image)
        self.last: tuple[int, int] | None = None

        root.title("MNIST 手写数字识别")
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        controls = tk.Frame(root)
        controls.pack(fill="x", pady=8)
        tk.Button(controls, text="识别", command=self.predict).pack(side="left", expand=True)
        tk.Button(controls, text="清空", command=self.clear).pack(side="left", expand=True)
        self.result = tk.Label(root, text="请在黑色区域写一个数字", font=("Arial", 16))
        self.result.pack(pady=(0, 8))

    def mouse_down(self, event: tk.Event) -> None:
        self.last = (event.x, event.y)

    def mouse_move(self, event: tk.Event) -> None:
        if self.last is None:
            return
        current = (event.x, event.y)
        self.canvas.create_line(*self.last, *current, fill="white", width=LINE_WIDTH, capstyle="round")
        self.drawer.line((*self.last, *current), fill=255, width=LINE_WIDTH)
        self.last = current

    def mouse_up(self, _event: tk.Event) -> None:
        self.last = None

    def clear(self) -> None:
        self.canvas.delete("all")
        self.drawer.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill=0)
        self.result.config(text="请在黑色区域写一个数字")

    def predict(self) -> None:
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        x = (torch.from_numpy(pixels) - MEAN) / STD
        with torch.no_grad():
            probabilities = F.softmax(self.model(x.unsqueeze(0)), dim=1)[0]
        prediction = int(probabilities.argmax())
        self.result.config(text=f"预测：{prediction}　置信度：{float(probabilities[prediction]):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="打开 MNIST 手写数字识别画板")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT_DIR / "simple_mlp.pt")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"找不到模型 {args.checkpoint}，请先运行 mnist_classifier.py")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        messagebox.showerror("启动失败", str(error))
        raise SystemExit(error) from error
    DigitCanvas(root, args.checkpoint)
    root.mainloop()


if __name__ == "__main__":
    main()
