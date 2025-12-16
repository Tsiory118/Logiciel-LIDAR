import sys
import os
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QGroupBox, QLabel, QFileDialog, QMessageBox
)
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ============================================================
#                  MODELE DE DONNEES (8x8)
# ============================================================

class RoadDataModel:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.Z = self._load_csv()

    def _load_csv(self) -> np.ndarray:
        try:
            data = np.genfromtxt(self.csv_file, delimiter=',', dtype=float)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            data = data[~np.isnan(data).all(axis=1)]  # supprimer lignes vides

            # Si moins de 8 lignes, compléter avec des zéros
            if data.shape[0] < 8:
                padding = np.zeros((8 - data.shape[0], data.shape[1]))
                data = np.vstack([padding, data])

            # Prendre les 8 dernières lignes et colonnes 1 à 8 (ignorer timestamp)
            data = data[-8:, 1:9]

            # Si moins de 8 colonnes, compléter avec des zéros
            if data.shape[1] < 8:
                padding = np.zeros((8, 8 - data.shape[1]))
                data = np.hstack([data, padding])

            return data.astype(float)
        except Exception as e:
            print(f"Warning: problème CSV : {e}")
            return np.zeros((8, 8))

# ============================================================
#                  CANVAS 3D
# ============================================================

class Surface3DCanvas(FigureCanvasQTAgg):
    def __init__(self, Z):
        self.figure = Figure(dpi=100)
        super().__init__(self.figure)

        self.ax = self.figure.add_subplot(111, projection="3d")
        self.X, self.Y = np.meshgrid(range(8), range(8))
        self.Z = Z

        self.elev, self.azim = 30, -60
        self.cmap = "viridis"
        self.cbar = None  # initialisation

        self.draw_surface()

    def draw_surface(self):
        self.ax.clear()

        self.surf = self.ax.plot_surface(
            self.X, self.Y, self.Z,
            cmap=self.cmap,
            edgecolor="black",
            linewidth=0.3
        )

        self.ax.view_init(self.elev, self.azim)
        self.ax.set_box_aspect((1, 1, 0.4))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_title("Carte 3D – État de la route")

        # Supprimer l'ancien colorbar en toute sécurité
        if hasattr(self, "cbar") and self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        # Créer un nouveau colorbar
        self.cbar = self.figure.colorbar(self.surf, shrink=0.6, pad=0.08)
        self.cbar.set_label("Déformation")

        self.draw_idle()

    def update_surface(self, Z):
        self.Z = Z
        self.draw_surface()

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
        self.draw_surface()

    def export_png(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Exporter PNG", "route.png", "Images (*.png)"
        )
        if fname:
            self.figure.savefig(fname, dpi=300)

# ============================================================
#                  CSV LIVE WATCHER
# ============================================================

class CSVLiveWatcher:
    def __init__(self, csv_path, canvas, status_label):
        self.csv_path = csv_path
        self.canvas = canvas
        self.status_label = status_label
        self.last_mtime = os.path.getmtime(csv_path)

        self.timer = QTimer()
        self.timer.timeout.connect(self.check)
        self.timer.start(500)

    def check(self):
        try:
            mtime = os.path.getmtime(self.csv_path)
            if mtime != self.last_mtime:
                self.last_mtime = mtime
                model = RoadDataModel(self.csv_path)
                self.canvas.update_surface(model.Z)
                t = datetime.now().strftime("%H:%M:%S")
                self.status_label.setText(f"{self.csv_path} | Mise à jour : {t}")
        except Exception as e:
            self.status_label.setText(f"Erreur CSV : {e}")

# ============================================================
#                  PANNEAU DE CONTROLE
# ============================================================

class ControlPanel(QGroupBox):
    def __init__(self, app):
        super().__init__("Contrôles")
        self.app = app
        self.canvas = app.canvas
        self.rotating = False

        layout = QVBoxLayout(self)

        # ---- DIRECTION ----
        grid = QGridLayout()
        directions = [
            ("↑", (0, 1), lambda: self.canvas.rotate(5, 0)),
            ("←", (1, 0), lambda: self.canvas.rotate(0, -5)),
            ("→", (1, 2), lambda: self.canvas.rotate(0, 5)),
            ("↓", (2, 1), lambda: self.canvas.rotate(-5, 0)),
        ]
        for txt, pos, action in directions:
            b = QPushButton(txt)
            b.clicked.connect(action)
            grid.addWidget(b, *pos)

        layout.addLayout(grid)

        # ---- IMPORT CSV (au-dessus de Rotation auto) ----
        btn_import = QPushButton("Importer CSV")
        btn_import.clicked.connect(self.app.import_csv)
        layout.addWidget(btn_import)

        # ---- ROTATION ----
        btn_auto = QPushButton("Rotation auto")
        btn_auto.clicked.connect(self.toggle_rotation)
        layout.addWidget(btn_auto)

        btn_reset = QPushButton("Réinitialiser vue")
        btn_reset.clicked.connect(self.canvas.reset_view)
        layout.addWidget(btn_reset)

        btn_export = QPushButton("Exporter PNG")
        btn_export.clicked.connect(self.canvas.export_png)
        layout.addWidget(btn_export)

        layout.addWidget(QLabel("Échelle de couleur"))
        cmap = QComboBox()
        cmap.addItems(["viridis", "plasma", "inferno", "cividis", "coolwarm"])
        cmap.currentTextChanged.connect(self.canvas.update_colormap)
        layout.addWidget(cmap)

        layout.addStretch()

    def toggle_rotation(self):
        self.rotating = not self.rotating
        if self.rotating:
            self.rotate_step()

    def rotate_step(self):
        if not self.rotating:
            return
        self.canvas.rotate(d_azim=1)
        QTimer.singleShot(120, self.rotate_step)

# ============================================================
#                  APPLICATION PRINCIPALE
# ============================================================

class RoadQualityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surveillance Qualité Route – 3D")
        self.resize(1100, 700)

        self.canvas = Surface3DCanvas(np.zeros((8, 8)))
        self.watcher = None

        self.status = QLabel("Aucun fichier CSV chargé")
        self.status.setStyleSheet("color:#455a64")

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Visualisation 3D – Qualité de Route</b>"))
        header.addStretch()

        main = QHBoxLayout()
        main.addWidget(self.canvas, 3)
        main.addWidget(ControlPanel(self), 1)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addLayout(main)
        layout.addWidget(self.status)

    def import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Importer CSV", "", "CSV (*.csv)"
        )
        if not fname:
            return

        model = RoadDataModel(fname)
        self.canvas.update_surface(model.Z)

        if self.watcher:
            self.watcher.timer.stop()

        self.watcher = CSVLiveWatcher(
            fname, self.canvas, self.status
        )

        self.status.setText(f"CSV chargé : {fname}")
        QMessageBox.information(
            self, "Import CSV", "CSV chargé avec succès ✅"
        )

# ============================================================
#                          MAIN
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
    QWidget { background:#f4f6f8; font-family:Segoe UI; }
    QPushButton {
        background:#1976d2; color:white; border-radius:6px; padding:6px;
    }
    QPushButton:hover { background:#1565c0; }
    QGroupBox { border:1px solid #cfd8dc; border-radius:8px; margin-top:10px; }
    """)

    window = RoadQualityApp()
    window.show()
    sys.exit(app.exec())
