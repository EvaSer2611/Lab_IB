
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
import subprocess, tempfile, os, threading, math


import matplotlib  # type: ignore
matplotlib.use("TkAgg")
from matplotlib.figure import Figure  # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore


def read_audio(path):

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", "44100", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        data, sr = sf.read(wav_path, always_2d=True)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    return data, sr


def write_audio(path, data, sr):

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    sf.write(wav_path, data, sr, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    os.remove(wav_path)


def play_audio(path):

    try:
        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Не найден ffplay. Установите FFmpeg и добавьте его в PATH.")


# ====== SCRAMBLER ALGORITHMS ======
def _pad_to_multiple(x, block_len):
    n = x.shape[0]
    rem = n % block_len
    if rem == 0:
        return x, 0
    pad = block_len - rem
    pad_shape = (pad, x.shape[1])
    return np.vstack([x, np.zeros(pad_shape, dtype=x.dtype)]), pad


def scramble_time(data, sr, T=1.0, n=10, seed=0, varying=False, unscramble=False):

    seg_len = int(round((T / n) * sr))
    if seg_len < 1:
        raise ValueError("Segment length < 1 sample. Увеличьте T или уменьшите n.")
    win_len = seg_len * n
    padded, pad = _pad_to_multiple(data, win_len)
    num_windows = padded.shape[0] // win_len
    rng = np.random.RandomState(seed)
    out = np.zeros_like(padded)
    keys = []

    if not varying:
        # single perm for all windows
        perm = rng.permutation(n)
        keys.append(perm.copy())
        inv = np.empty_like(perm)
        inv[perm] = np.arange(n)
        for i in range(num_windows):
            start = i * win_len
            for j in range(n):
                s = start + j * seg_len
                d = start + (perm[j] if not unscramble else inv[j]) * seg_len
                out[d:d+seg_len] = padded[s:s+seg_len]
    else:
        # different perm each window
        for i in range(num_windows):
            perm = rng.permutation(n)
            keys.append(perm.copy())
            inv = np.empty_like(perm)
            inv[perm] = np.arange(n)
            start = i * win_len
            for j in range(n):
                s = start + j * seg_len
                d = start + (perm[j] if not unscramble else inv[j]) * seg_len
                out[d:d+seg_len] = padded[s:s+seg_len]

    if pad:
        out = out[:-pad]
    return out, keys


def scramble_freq(data, sr, T=1.0, n=5, seed=0, varying=False, unscramble=False):
    
    win_len = int(round(T * sr))
    if win_len < 1:
        raise ValueError("Window length < 1 sample. Увеличьте T.")
    padded, pad = _pad_to_multiple(data, win_len)
    num_windows = padded.shape[0] // win_len
    rng = np.random.RandomState(seed)
    out = np.zeros_like(padded)
    keys = []

    if not varying:
        perm = rng.permutation(n)
        keys.append(perm.copy())
        inv = np.empty_like(perm)
        inv[perm] = np.arange(n)
        for w in range(num_windows):
            start = w * win_len
            frame = padded[start:start+win_len, 0]
            spec = np.fft.rfft(frame)
            B = len(spec)
            sizes = [B // n] * n
            for i in range(B % n):
                sizes[i] += 1
            idx = 0
            bands = []
            for s in sizes:
                bands.append(spec[idx:idx+s])
                idx += s
            new_spec = np.concatenate([bands[perm[i] if not unscramble else inv[i]] for i in range(n)])
            out[start:start+win_len, 0] = np.fft.irfft(new_spec, n=win_len)
    else:
        for w in range(num_windows):
            perm = rng.permutation(n)
            keys.append(perm.copy())
            inv = np.empty_like(perm)
            inv[perm] = np.arange(n)
            start = w * win_len
            frame = padded[start:start+win_len, 0]
            spec = np.fft.rfft(frame)
            B = len(spec)
            sizes = [B // n] * n
            for i in range(B % n):
                sizes[i] += 1
            idx = 0
            bands = []
            for s in sizes:
                bands.append(spec[idx:idx+s])
                idx += s
            new_spec = np.concatenate([bands[perm[i] if not unscramble else inv[i]] for i in range(n)])
            out[start:start+win_len, 0] = np.fft.irfft(new_spec, n=win_len)

    if pad:
        out = out[:-pad]
    return out, keys


# ====== PLOTTING HELPERS ======
def downsample_for_plot(y, max_points=200_000):
    L = y.shape[0]
    if L <= max_points:
        return y
    factor = math.ceil(L / max_points)
    trimmed_len = (L // factor) * factor
    y = y[:trimmed_len]
    return y.reshape(-1, factor).mean(axis=1)


def plot_wave_on_axis(ax, data, sr, title=""):
    ax.clear()
    y = data[:, 0]
    y_ds = downsample_for_plot(y)
    t = np.arange(y_ds.shape[0]) * (len(y) / y_ds.shape[0]) / float(sr)
    ax.plot(t, y_ds)
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True)


# ====== GUI ======
class ScramblerApp:
    def __init__(self, root):
        self.keys = []
        self.root = root
        root.title("Аудио-скремблёр 🎧 + графики (fixed)")
        root.geometry("900x700")
        root.resizable(True, True)

        # single key label (no duplicates)
        self.key_label = tk.Label(root, text="Ключ: —", fg="green", font=("Arial", 10))
        self.key_label.pack(anchor="w", padx=8, pady=3)

        # vars
        self.infile = tk.StringVar()
        self.outfile = None
        self.mode = tk.StringVar(value="time")
        self.T = tk.DoubleVar(value=1.0)
        self.n = tk.IntVar(value=10)
        self.seed = tk.IntVar(value=42)
        self.varying = tk.BooleanVar()
        self.unscramble = tk.BooleanVar()

        self.orig_data = None
        self.orig_sr = None
        self.result_data = None
        self.result_sr = None

        # top controls
        ctl_frame = tk.Frame(root)
        ctl_frame.pack(anchor="nw", fill="x", padx=8, pady=6)

        tk.Label(ctl_frame, text="Входной файл:").grid(row=0, column=0, sticky="w")
        tk.Entry(ctl_frame, textvariable=self.infile, width=60).grid(row=0, column=1, columnspan=4, sticky="w")
        tk.Button(ctl_frame, text="Выбрать...", command=self.choose_file).grid(row=0, column=5, padx=6)

        tk.Button(ctl_frame, text="▶ Прослушать исходный", command=self.play_original).grid(row=1, column=0, pady=6)
        tk.Button(ctl_frame, text="▶ Прослушать результат", command=self.play_result).grid(row=1, column=1, pady=6)

        tk.Label(ctl_frame, text="Режим:").grid(row=2, column=0, sticky="w")
        tk.Radiobutton(ctl_frame, text="Временной (time)", variable=self.mode, value="time").grid(row=2, column=1, sticky="w")
        tk.Radiobutton(ctl_frame, text="Частотный (freq)", variable=self.mode, value="freq").grid(row=2, column=2, sticky="w")

        tk.Label(ctl_frame, text="T (сек):").grid(row=3, column=0, sticky="w")
        tk.Entry(ctl_frame, textvariable=self.T, width=8).grid(row=3, column=1, sticky="w")
        tk.Label(ctl_frame, text="n (сегментов):").grid(row=3, column=2, sticky="w")
        tk.Entry(ctl_frame, textvariable=self.n, width=8).grid(row=3, column=3, sticky="w")
        tk.Label(ctl_frame, text="Seed:").grid(row=3, column=4, sticky="w")
        tk.Entry(ctl_frame, textvariable=self.seed, width=8).grid(row=3, column=5, sticky="w")

        tk.Checkbutton(ctl_frame, text="Режим расшифровки", variable=self.unscramble).grid(row=4, column=3, columnspan=3, sticky="w")

        tk.Button(ctl_frame, text="▶ Запустить скремблирование", command=self.run_scramble).grid(row=5, column=0, columnspan=2, pady=8)

        self.status = tk.Label(root, text="", fg="blue")
        self.status.pack(anchor="w", padx=8, pady=4)

        # plotting area: two plots side by side
        plot_frame = tk.Frame(root)
        plot_frame.pack(fill="both", expand=True, padx=8, pady=6)

        # Original plot
        fig1 = Figure(figsize=(5, 3))
        self.ax_orig = fig1.add_subplot(111)
        self.canvas_orig = FigureCanvasTkAgg(fig1, master=plot_frame)
        self.canvas_orig.get_tk_widget().pack(side="left", fill="both", expand=True)

        # Result plot
        fig2 = Figure(figsize=(5, 3))
        self.ax_res = fig2.add_subplot(111)
        self.canvas_res = FigureCanvasTkAgg(fig2, master=plot_frame)
        self.canvas_res.get_tk_widget().pack(side="right", fill="both", expand=True)

    def choose_file(self):
        path = filedialog.askopenfilename(filetypes=[("Аудио", "*.mp3 *.wav *.flac *.ogg")])
        if path:
            self.infile.set(path)
            # try to read and plot original immediately (in background)
            threading.Thread(target=self._load_and_plot_original, args=(path,)).start()

    def _load_and_plot_original(self, path):
        try:
            self.status.config(text="Загрузка и построение графика исходного...")
            data, sr = read_audio(path)
            self.orig_data = data
            self.orig_sr = sr
            # plot on main thread
            self.root.after(0, lambda: self._plot_original())
            self.status.config(text="Исходный загружен.")
        except Exception as e:
            self.status.config(text="Ошибка при загрузке.")
            messagebox.showerror("Ошибка", str(e))

    def _plot_original(self):
        if self.orig_data is None:
            return
        plot_wave_on_axis(self.ax_orig, self.orig_data, self.orig_sr, title="Оригинал")
        self.canvas_orig.draw()

    def _plot_result(self):
        if self.result_data is None:
            return
        plot_wave_on_axis(self.ax_res, self.result_data, self.result_sr, title="Результат")
        self.canvas_res.draw()

    def play_original(self):
        if not self.infile.get():
            messagebox.showerror("Ошибка", "Выберите файл.")
            return
        play_audio(self.infile.get())

    def play_result(self):
        if self.outfile and os.path.exists(self.outfile):
            play_audio(self.outfile)
        elif self.result_data is not None:
            # temporarily write result to temp file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                write_audio(tmp_path, self.result_data, self.result_sr)
                play_audio(tmp_path)
            finally:
                pass
        else:
            messagebox.showerror("Ошибка", "Сначала выполните скремблирование.")

    def run_scramble(self):
        path = self.infile.get()
        if not path:
            messagebox.showerror("Ошибка", "Выберите аудиофайл.")
            return
        outfile = filedialog.asksaveasfilename(defaultextension=".mp3",
                                               filetypes=[("MP3", "*.mp3"), ("WAV", "*.wav")])
        if not outfile:
            return
        self.status.config(text="Обработка...")
        threading.Thread(target=self.process, args=(path, outfile)).start()

    def process(self, path, outfile):
        try:
            # If we already loaded original, reuse; otherwise load now
            if self.orig_data is None or self.infile.get() != path:
                data, sr = read_audio(path)
            else:
                data, sr = self.orig_data, self.orig_sr

            if self.mode.get() == "time":
                result, keys = scramble_time(data, sr, self.T.get(), self.n.get(),
                                             self.seed.get(), self.varying.get(), self.unscramble.get())
            else:
                result, keys = scramble_freq(data, sr, self.T.get(), self.n.get(),
                                             self.seed.get(), self.varying.get(), self.unscramble.get())

            write_audio(outfile, result, sr)
            self.outfile = outfile
            self.result_data = result
            self.result_sr = sr
            self.keys = keys

            # формируем строку ключа (перестановок)
            if self.varying.get():
                key_lines = [f"окно {i+1}: {k.tolist()}" for i, k in enumerate(self.keys)]
                key_str = "Ключи (по окнам):\n" + "\n".join(key_lines)
            else:
                key_str = f"Ключ (fixed): {self.keys[0].tolist() if len(self.keys)>0 else '—'}"

            self.root.after(0, lambda: self.key_label.config(text=key_str))

            # update plots on main thread
            self.root.after(0, lambda: self._plot_result())

            self.status.config(text=f"✅ Готово: {os.path.basename(outfile)}")
            messagebox.showinfo("Готово", f"Файл сохранён:\n{outfile}")
        except Exception as e:
            self.status.config(text="❌ Ошибка")
            messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    ScramblerApp(root)
    root.mainloop()
