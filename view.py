import sys
import os
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QGroupBox, QLabel, QFileDialog,
    QMessageBox, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import QTimer, Qt
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

            cols_to_take = min(data.shape[1] - 1, 8)
            data = data[-8:, 1:1 + cols_to_take]

            if data.shape[1] < 8:
                padding = np.zeros((8, 8 - data.shape[1]))
                data = np.hstack([data, padding])

            return data.astype(float)

        except Exception as e:
            print(f"Erreur CSV : {e}")
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
            edgecolor='none',
            antialiased=True
        )

        self.ax.view_init(self.elev, self.azim)
        self.ax.set_box_aspect((1, 1, 0.4))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_title("Carte 3D – État de la route", fontsize=12, weight="bold", pad=15)

        if self.cbar:
            try:
                self.cbar.remove()
            except Exception:
                pass

        self.cbar = self.figure.colorbar(self.surf, shrink=0.6, pad=0.08)
        self.cbar.set_label("Déformation (cm)", fontsize=10)

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
            self,
            "Exporter PNG",
            f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "Images (*.png)"
        )
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
        self.timer.start(300)

    def check(self):
        try:
            mtime = os.path.getmtime(self.csv_path)
            if mtime != self.last_mtime:
                self.last_mtime = mtime
                model = RoadDataModel(self.csv_path)
                self.canvas.update_surface(model.Z)
                self.update_analysis(model.Z)
                self.status_label.setText(
                    f"{self.csv_path} | Mise à jour : {datetime.now().strftime('%H:%M:%S')}"
                )
        except Exception as e:
            self.status_label.setText(f"Erreur CSV : {e}")

    def update_analysis(self, Z):
        # Conversion mm → cm
        avg_val = np.round(np.mean(Z) / 10, 2)
        max_val = np.round(np.max(Z) / 10, 2)
        min_val = np.round(np.min(Z) / 10, 2)

        text = (
            "<h3 style='color:black;'>Analyse de la qualité de la route</h3>"
            "<hr>"
            "<p><b>Déformation moyenne :</b><br>{} cm</p>"
            "<p><b>Déformation maximale :</b><br>{} cm</p>"
            "<p><b>Déformation minimale :</b><br>{} cm</p>"
            "<p style='font-size:12px;'><b>Échelle :</b> 1 unité = 1 cm</p>"
        ).format(avg_val, max_val, min_val)

        self.analysis_label.setText(text)


# ==================== PANNEAU DE CONTROLE ====================
class ControlPanel(QGroupBox):
    def __init__(self, app):
        super().__init__("Contrôles")
        self.app = app
        self.canvas = app.canvas
        self.rotating = False

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        grid = QGridLayout()
        buttons = [
            ("↑", (0, 1), lambda: self.set_view(elev=90)),
            ("←", (1, 0), lambda: self.set_view(azim=self.canvas.azim - 15)),
            ("→", (1, 2), lambda: self.set_view(azim=self.canvas.azim + 15)),
            ("↓", (2, 1), lambda: self.set_view(elev=0)),
        ]

        for txt, pos, action in buttons:
            btn = QPushButton(txt)
            btn.setMinimumSize(40, 40)
            btn.clicked.connect(action)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)  # <-- CURSEUR POINTEUR
            grid.addWidget(btn, *pos)

        layout.addLayout(grid)

        btn_import = QPushButton("Importer CSV")
        btn_import.clicked.connect(self.app.import_csv)
        btn_import.setCursor(Qt.CursorShape.PointingHandCursor)  # <-- CURSEUR POINTEUR
        layout.addWidget(btn_import)

        btn_auto = QPushButton("Rotation auto")
        btn_auto.clicked.connect(self.toggle_rotation)
        btn_auto.setCursor(Qt.CursorShape.PointingHandCursor)  # <-- CURSEUR POINTEUR
        layout.addWidget(btn_auto)

        btn_reset = QPushButton("Réinitialiser vue")
        btn_reset.clicked.connect(self.canvas.reset_view)
        btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)  # <-- CURSEUR POINTEUR
        layout.addWidget(btn_reset)

        btn_export = QPushButton("Exporter PNG")
        btn_export.clicked.connect(self.canvas.export_png)
        btn_export.setCursor(Qt.CursorShape.PointingHandCursor)  # <-- CURSEUR POINTEUR
        layout.addWidget(btn_export)

        self.analysis_label = QLabel("Aucune donnée CSV")
        self.analysis_label.setStyleSheet(
            "background:#ffffff; color:#000; padding:8px; border-radius:6px;"
        )
        self.analysis_label.setWordWrap(True)

        layout.addWidget(self.analysis_label)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def set_view(self, elev=None, azim=None):
        if elev is not None:
            self.canvas.elev = elev
        if azim is not None:
            self.canvas.azim = azim
        self.canvas.ax.view_init(self.canvas.elev, self.canvas.azim)
        self.canvas.draw_idle()

    def toggle_rotation(self):
        self.rotating = not self.rotating
        if self.rotating:
            self.rotate_step()

    def rotate_step(self):
        if not self.rotating:
            return
        self.canvas.rotate(d_azim=0.5)
        QTimer.singleShot(30, self.rotate_step)


# ==================== APPLICATION PRINCIPALE ====================
class RoadQualityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surveillance de la qualité de Route")
        self.resize(1200, 700)

        self.canvas = Surface3DCanvas(np.zeros((8, 8)))
        self.watcher = None

        self.status = QLabel("Aucun fichier CSV chargé")

        main = QHBoxLayout()
        self.control_panel = ControlPanel(self)
        main.addWidget(self.canvas, 3)
        main.addWidget(self.control_panel, 1)

        layout = QVBoxLayout(self)
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

        self.watcher = CSVLiveWatcher(
            fname, self.canvas, self.status, self.control_panel.analysis_label
        )
        self.watcher.update_analysis(model.Z)

        self.status.setText(f"CSV chargé : {fname}")
        QMessageBox.information(self, "Import CSV", "Importation réussie ✅")


# ==================== MAIN ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QWidget { background:#f4f6f8; font-family:Segoe UI; }
        QPushButton {
            background:#1976d2; color:white; border-radius:8px;
            padding:8px; font-weight:bold;
        }
        QPushButton:hover { background:#1565c0; }
        QGroupBox { border:1px solid #cfd8dc; border-radius:8px; }
    """)

    window = RoadQualityApp()
    window.show()
    sys.exit(app.exec())
