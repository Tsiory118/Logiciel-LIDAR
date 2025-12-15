import sys
import os
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QGroupBox, QLabel, QFileDialog, QMessageBox
)
from PySide6.QtCore import QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ============================================================
#                  MODELE DE DONNEES (CSV LIVE)
# ============================================================

class RoadDataModel:
    """Lecture CSV live pour matrice 8x8"""
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.last_mtime = 0
        self.Z = self._load_csv()

    def _load_csv(self) -> np.ndarray:
        raw = np.genfromtxt(self.csv_file, delimiter=',', skip_header=1)

        if raw.ndim == 2 and raw.shape[1] > 8:
            raw = raw[:, -8:]
        if raw.ndim == 2 and raw.shape[0] >= 8:
            raw = raw[-8:, :]
        if raw.ndim == 1 and raw.size == 64:
            raw = raw.reshape((8, 8))

        if raw.shape != (8, 8):
            raise ValueError(f"Dimensions invalides {raw.shape} (attendu 8x8)")
        return raw

    def reload_if_changed(self) -> bool:
        try:
            mtime = os.path.getmtime(self.csv_file)
            if mtime != self.last_mtime:
                self.last_mtime = mtime
                self.Z = self._load_csv()
                return True
        except Exception:
            pass
        return False

# ============================================================
#                  CANVAS MATPLOTLIB 3D
# ============================================================

class Surface3DCanvas(FigureCanvasQTAgg):
    def __init__(self, Z: np.ndarray):
        self.figure = Figure(facecolor="white", dpi=100)
        super().__init__(self.figure)

        self.ax = self.figure.add_subplot(111, projection="3d")
        self.X, self.Y = np.meshgrid(range(8), range(8))

        self.Z = Z
        self.cmap = "viridis"
        self.elev, self.azim = 30, -60

        self._draw_surface()

    def _draw_surface(self):
        self.ax.clear()

        self.surf = self.ax.plot_surface(
            self.X, self.Y, self.Z,
            cmap=self.cmap,
            edgecolor="#666",
            linewidth=0.25
        )

        self.ax.view_init(self.elev, self.azim)
        self.ax.set_box_aspect((1, 1, 0.4))

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_title("Carte 3D – État de la route", pad=15)

        if hasattr(self, "cbar"):
            self.cbar.remove()

        self.cbar = self.figure.colorbar(self.surf, ax=self.ax, shrink=0.55)
        self.cbar.set_label("Déformation (mm)")

        self.draw_idle()

    def update_surface(self, Z):
        self.Z = Z
        self.surf.remove()
        self.surf = self.ax.plot_surface(
            self.X, self.Y, self.Z,
            cmap=self.cmap,
            edgecolor="#666",
            linewidth=0.25
        )
        self.cbar.update_normal(self.surf)
        self.draw_idle()

    def rotate(self, d_elev=0, d_azim=0):
        self.elev += d_elev
        self.azim += d_azim
        self.ax.view_init(self.elev, self.azim)
        self.draw_idle()

    def reset_view(self):
        self.elev, self.azim = 30, -60
        self.ax.view_init(self.elev, self.azim)
        self.draw_idle()

    def update_colormap(self, cmap):
        self.cmap = cmap
        self.surf.set_cmap(cmap)
        self.cbar.update_normal(self.surf)
        self.draw_idle()

    def export_png(self):
        fname, _ = QFileDialog.getSaveFileName(
            None, "Exporter PNG", "route_3d.png", "Images (*.png)"
        )
        if fname:
            self.figure.savefig(fname, dpi=300)

# ============================================================
#                  PANNEAU DE CONTROLE
# ============================================================

class ControlPanel(QGroupBox):
    def __init__(self, canvas: Surface3DCanvas, app):
        super().__init__("Contrôle")
        self.canvas = canvas
        self.app = app
        self.rotating = False

        layout = QHBoxLayout(self)

        arrows = QGridLayout()
        for (txt, r, c, f) in [
            ("↑", 0, 1, lambda: canvas.rotate(5, 0)),
            ("←", 1, 0, lambda: canvas.rotate(0, -5)),
            ("→", 1, 2, lambda: canvas.rotate(0, 5)),
            ("↓", 2, 1, lambda: canvas.rotate(-5, 0)),
        ]:
            btn = QPushButton(txt)
            btn.clicked.connect(f)
            btn.setMinimumSize(45, 45)
            arrows.addWidget(btn, r, c)

        layout.addLayout(arrows)

        actions = QVBoxLayout()
        btn_import = QPushButton("Importer CSV")
        btn_import.clicked.connect(self.import_csv)

        btn_export = QPushButton("Exporter PNG")
        btn_export.clicked.connect(canvas.export_png)

        btn_reset = QPushButton("Réinitialiser vue")
        btn_reset.clicked.connect(canvas.reset_view)

        actions.addWidget(btn_import)
        actions.addWidget(btn_export)
        actions.addWidget(btn_reset)

        actions.addWidget(QLabel("Colormap"))
        cmap = QComboBox()
        cmap.addItems(["viridis", "plasma", "inferno", "coolwarm"])
        cmap.currentTextChanged.connect(canvas.update_colormap)
        actions.addWidget(cmap)

        actions.addStretch()
        layout.addLayout(actions)

    def import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(
            None, "Choisir un CSV", "", "CSV (*.csv)"
        )
        if fname:
            self.app.model = RoadDataModel(fname)
            self.canvas.update_surface(self.app.model.Z)
            QMessageBox.information(self, "CSV", "Surveillance LIVE activée")

# ============================================================
#                  APPLICATION PRINCIPALE
# ============================================================

class RoadQualityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surveillance Qualité Route – 3D LIVE")
        self.resize(1000, 650)

        self.model = None
        self.canvas = Surface3DCanvas(np.zeros((8, 8)))
        self.controls = ControlPanel(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.controls)

        # ⏱️ Timer LIVE CSV
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_from_csv)
        self.timer.start(500)  # 500 ms

    def update_from_csv(self):
        if self.model and self.model.reload_if_changed():
            self.canvas.update_surface(self.model.Z)

# ============================================================
#                          MAIN
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RoadQualityApp()
    window.show()
    sys.exit(app.exec())
