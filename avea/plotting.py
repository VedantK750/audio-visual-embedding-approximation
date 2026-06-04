"""Live training-curve plot shared by the trainers."""

import os


class LivePlot:

    def __init__(self, enabled=True, save_path="training_curve.png"):
        self.enabled = enabled
        self.save_path = os.path.abspath(save_path)
        self.train_total = []
        self.val_total = []
        self.plt = None
        self.fig = None
        self.ax = None
        self.interactive = False
        if not self.enabled:
            return
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            # An interactive backend needs a display; with Agg (headless) we
            # skip live drawing and rely entirely on savefig.
            self.interactive = "agg" not in matplotlib.get_backend().lower()
            self.plt = plt
            if self.interactive:
                plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 5))
        except Exception as e:
            print(f"[LivePlot] matplotlib unavailable ({e}); plotting disabled")
            self.enabled = False

    def update(self, train_metrics, val_metrics):
        self.train_total.append(train_metrics["total"])
        self.val_total.append(val_metrics["total"])
        if not self.enabled:
            return
        self._render()
        # Write the PNG first, so it always exists regardless of the backend.
        try:
            self.fig.savefig(self.save_path)
            if len(self.train_total) == 1:
                print(f"[LivePlot] saving curve to {self.save_path}")
        except Exception as e:
            print(f"[LivePlot] could not save figure: {e}")
        # Best-effort live refresh; a display problem must never crash training.
        if self.interactive:
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                self.plt.pause(0.01)
            except Exception:
                self.interactive = False

    def _render(self):
        epochs = range(1, len(self.train_total) + 1)
        self.ax.clear()
        self.ax.plot(epochs, self.train_total, "-o", label="Train", color="tab:blue")
        self.ax.plot(epochs, self.val_total, "-o", label="Val", color="tab:orange")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Total loss")
        self.ax.set_title("Training Progress")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

    def close(self):
        if not self.enabled:
            return
        try:
            self.fig.savefig(self.save_path)
        except Exception:
            pass
        if self.interactive:
            try:
                self.plt.ioff()
                self.plt.show()
            except Exception:
                pass
