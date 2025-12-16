import sys
import os
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QGroupBox, QLabel, QFileDialog, QMessageBox, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCursor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ==================== MODELE DE DONNEES ====================
class RoadDataModel:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.Z = self._load_csv()

    def _load_csv(self) -> np.ndarray:
        try:
            data = np.genfromtxt(self.csv_file, delimiter=',', dtype=float)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            data = data[~np.isnan(data).all(axis=1)]
            if data.shape[0] < 8:
                padding = np.zeros((8 - data.shape[0], data.shape[1]))
                data = np.vstack([padding, data])
            data = data[-8:, 1:9]  # ignorer timestamp, garder 8 colonnes
            if data.shape[1] < 8:
                padding = np.zeros((8, 8 - data.shape[1]))
                data = np.hstack([data, padding])
            return data.astype(float)
        except Exception as e:
            print(f"Warning: problème CSV : {e}")
            return np.zeros((8, 8))

# ==================== CANVAS 3D ====================
class Surface3DCanvas(FigureCanvasQTAgg):
    def __init__(self, Z):
        self.figure = Figure(dpi=100)
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.X, self.Y = np.meshgrid(range(8), range(8))
        self.Z = Z
        self.elev, self.azim = 30, -60
        self.cmap = "viridis"
        self.cbar = None
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
        self.ax.set_title("Carte 3D – État de la route", pad=15, fontsize=12, weight="bold")
        if hasattr(self, "cbar") and self.cbar is not None:
            try:
                self.cbar.remove()
            except:
                pass
            self.cbar = None
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
        fname, _ = QFileDialog.getSaveFileName(self, "Exporter PNG", "route.png", "Images (*.png)")
        if fname:
            self.figure.savefig(fname, dpi=300)

# ==================== CSV LIVE WATCHER ====================
class CSVLiveWatcher:
    def __init__(self, csv_path, canvas, status_label, analysis_label):
        self.csv_path = csv_path
        self.canvas = canvas
        self.status_label = status_label
        self.analysis_label = analysis_label
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
                self.update_analysis(model.Z)
        except Exception as e:
            self.status_label.setText(f"Erreur CSV : {e}")

    def update_analysis(self, Z):
        avg_val = np.round(np.mean(Z), 2)
        max_val = np.max(Z)
        min_val = np.min(Z)
        text = (
            f"<b>Analyse de la qualité de la route :</b><br>"
            f"- Déformation moyenne : {avg_val} mm<br>"
            f"- Déformation max : {max_val} mm<br>"
            f"- Déformation min : {min_val} mm<br>"
            f"- Échelle : 1 unité = 1 cm²"
        )
        self.analysis_label.setText(text)

# ==================== PANNEAU DE CONTROLE ====================
class ControlPanel(QGroupBox):
    def __init__(self, app):
        super().__init__("Contrôles")
        self.app = app
        self.canvas = app.canvas
        self.rotating = False

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

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
            b.setMinimumSize(40, 40)
            b.clicked.connect(action)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            grid.addWidget(b, *pos)
        layout.addLayout(grid)
        layout.addSpacing(10)

        # ---- IMPORT CSV ----
        btn_import = QPushButton("Importer CSV")
        btn_import.setMinimumHeight(40)
        btn_import.clicked.connect(self.app.import_csv)
        btn_import.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(btn_import)
        layout.addSpacing(10)

        # ---- ROTATION / RESET / EXPORT ----
        btn_auto = QPushButton("Rotation auto")
        btn_auto.setMinimumHeight(35)
        btn_auto.clicked.connect(self.toggle_rotation)
        btn_auto.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(btn_auto)

        btn_reset = QPushButton("Réinitialiser vue")
        btn_reset.setMinimumHeight(35)
        btn_reset.clicked.connect(self.canvas.reset_view)
        btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(btn_reset)

        btn_export = QPushButton("Exporter PNG")
        btn_export.setMinimumHeight(35)
        btn_export.clicked.connect(self.canvas.export_png)
        btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(btn_export)
        layout.addSpacing(15)

        # ---- COLOREMAP ----
        layout.addWidget(QLabel("Échelle de couleur"))
        cmap = QComboBox()
        cmap.addItems(["viridis", "plasma", "inferno", "cividis", "coolwarm"])
        cmap.currentTextChanged.connect(self.canvas.update_colormap)
        layout.addWidget(cmap)
        layout.addSpacing(10)

        # ---- ANALYSE DE LA QUALITE DE ROUTE ----
        self.analysis_label = QLabel("Aucune donnée CSV")
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setStyleSheet(
            "background:#e0e0e0; border-radius:6px; padding:8px;"
        )
        layout.addWidget(self.analysis_label)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    # ---- Méthodes rotation ----
    def toggle_rotation(self):
        self.rotating = not self.rotating
        if self.rotating:
            self.rotate_step()

    def rotate_step(self):
        if not self.rotating:
            return
        self.canvas.rotate(d_azim=1)
        QTimer.singleShot(120, self.rotate_step)

# ==================== APPLICATION PRINCIPALE ====================
class RoadQualityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surveillance Qualité Route – 3D")
        self.resize(1200, 700)
        self.canvas = Surface3DCanvas(np.zeros((8, 8)))
        self.watcher = None

        self.status = QLabel("Aucun fichier CSV chargé")
        self.status.setStyleSheet("color:#455a64;")

        header = QHBoxLayout()
        title = QLabel("<b>Visualisation 3D – Qualité de Route</b>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(title)
        header.addStretch()

        main = QHBoxLayout()
        self.control_panel = ControlPanel(self)
        main.addWidget(self.canvas, 3)
        main.addWidget(self.control_panel, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addLayout(main)
        layout.addWidget(self.status)

    def import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Importer CSV", "", "CSV (*.csv)")
        if not fname:
            return
        model = RoadDataModel(fname)
        self.canvas.update_surface(model.Z)
        if self.watcher:
            self.watcher.timer.stop()
        self.watcher = CSVLiveWatcher(fname, self.canvas, self.status, self.control_panel.analysis_label)
        self.watcher.update_analysis(model.Z)
        self.status.setText(f"CSV chargé : {fname}")
        QMessageBox.information(self, "Import CSV", "Importation CSV réussie ✅")

# ==================== MAIN ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
    QWidget { background:#f4f6f8; font-family:Segoe UI; }
    QPushButton {
        background:#1976d2; color:white; border-radius:6px; padding:8px;
        font-weight:bold;
    }
    QPushButton:hover { background:#1565c0; }
    QGroupBox { border:1px solid #cfd8dc; border-radius:8px; margin-top:10px; padding:10px; }
    QComboBox { padding:4px; }
    QLabel { font-size:12px; }
    """)

    window = RoadQualityApp()
    window.show()
    sys.exit(app.exec())
