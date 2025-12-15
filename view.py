import sys
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
#                  MODELE DE DONNEES
# ============================================================

class RoadDataModel:
    """Lecture CSV Nx8 pour visualisation 3D 8x8."""
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.Z = self._load_csv()

    def _load_csv(self) -> np.ndarray:
        try:
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
        except Exception as e:
            QMessageBox.critical(None, "Erreur CSV", f"Impossible de lire '{self.csv_file}':\n{e}")
            sys.exit(1)

# ============================================================
#                  CANVAS MATPLOTLIB 3D
# ============================================================

class Surface3DCanvas(FigureCanvasQTAgg):
    """Surface 3D Matplotlib"""
    def __init__(self, Z: np.ndarray):
        self.figure = Figure(facecolor="white", dpi=100)
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.Z = Z
        self.X, self.Y = np.meshgrid(range(8), range(8))

        self.default_elev = 30
        self.default_azim = -60
        self.default_aspect = (1, 1, 0.4)

        self.elev = self.default_elev
        self.azim = self.default_azim
        self.cmap = "viridis"

        self._draw_surface()

    def _draw_surface(self):
        self.ax.clear()
        self.surf = self.ax.plot_surface(
            self.X, self.Y, self.Z,
            cmap=self.cmap,
            edgecolor="#666666",
            linewidth=0.2,
            antialiased=True
        )
        self.ax.view_init(self.elev, self.azim)
        self.ax.set_box_aspect(self.default_aspect)

        self.ax.set_xticks(range(8))
        self.ax.set_yticks(range(8))
        self.ax.set_zticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

        self.ax.set_title("Carte 3D – État de la route", fontsize=12, pad=18)

        if hasattr(self, "cbar"):
            self.cbar.remove()
        self.cbar = self.figure.colorbar(self.surf, ax=self.ax, shrink=0.55, pad=0.08)
        self.cbar.set_label("Déformation (mm)", fontsize=9)

        self.draw_scale_box(length=0.8)
        self.draw_cage()
        self.draw_idle()

    def update_surface(self, Z: np.ndarray):
        self.Z = Z
        self.surf.remove()
        self.surf = self.ax.plot_surface(
            self.X, self.Y, self.Z,
            cmap=self.cmap,
            edgecolor="#666666",
            linewidth=0.2,
            antialiased=True
        )
        self.cbar.update_normal(self.surf)
        self.draw_scale_box(length=0.8)
        self.draw_cage()
        self.draw_idle()

    def rotate(self, d_elev=0, d_azim=0):
        self.elev += d_elev
        self.azim += d_azim
        self.ax.view_init(self.elev, self.azim)
        self.draw_idle()

    def update_colormap(self, cmap_name):
        self.cmap = cmap_name
        self.surf.set_cmap(cmap_name)
        self.cbar.update_normal(self.surf)
        self.draw_idle()

    def reset_view(self):
        self.elev = self.default_elev
        self.azim = self.default_azim
        self.ax.set_box_aspect(self.default_aspect)
        self.ax.view_init(self.elev, self.azim)
        self.draw_idle()

    def export_png(self):
        fname, _ = QFileDialog.getSaveFileName(
            None, "Exporter PNG", "road_snapshot.png", "Images (*.png)"
        )
        if fname:
            self.figure.savefig(fname, dpi=300)
            print(f"Exporté : {fname}")

    def draw_scale_box(self, length=0.8):
        x0, y0 = np.min(self.X), np.min(self.Y)
        z0 = np.min(self.Z)
        xx = [x0, x0+length, x0+length, x0, x0, x0+length, x0+length, x0]
        yy = [y0, y0, y0+length, y0+length, y0, y0, y0+length, y0+length]
        zz = [z0, z0, z0, z0, z0+length, z0+length, z0+length, z0+length]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            self.ax.plot([xx[e[0]], xx[e[1]]],
                         [yy[e[0]], yy[e[1]]],
                         [zz[e[0]], zz[e[1]]],
                         color='black', linewidth=1)

    def draw_cage(self):
        for i in range(9):
            self.ax.plot([0,7],[i,i],[0,0], color='grey', alpha=0.3)
            self.ax.plot([i,i],[0,7],[0,0], color='grey', alpha=0.3)

# ============================================================
#                  PANNEAU DE CONTROLE
# ============================================================

class ControlPanel(QGroupBox):
    """Boutons et contrôle de la vue"""
    def __init__(self, canvas: Surface3DCanvas):
        super().__init__("Contrôle de la vue")
        self.canvas = canvas
        self.rotating = False
        self.rotation_speed = 1  # rotation lente

        main_layout = QHBoxLayout(self)

        # --- Colonne gauche : flèches style manette ---
        arrow_layout = QGridLayout()
        btn_up = QPushButton("↑")
        btn_left = QPushButton("←")
        btn_right = QPushButton("→")
        btn_down = QPushButton("↓")

        btn_up.clicked.connect(lambda: canvas.rotate(d_elev=5))
        btn_down.clicked.connect(lambda: canvas.rotate(d_elev=-5))
        btn_left.clicked.connect(lambda: canvas.rotate(d_azim=-5))
        btn_right.clicked.connect(lambda: canvas.rotate(d_azim=5))

        arrow_layout.addWidget(btn_up, 0, 1)
        arrow_layout.addWidget(btn_left, 1, 0)
        arrow_layout.addWidget(btn_right, 1, 2)
        arrow_layout.addWidget(btn_down, 2, 1)

        arrow_layout.setContentsMargins(10,10,10,10)
        arrow_layout.setHorizontalSpacing(10)
        arrow_layout.setVerticalSpacing(10)
        main_layout.addLayout(arrow_layout, 1)

        # --- Colonne droite : actions ---
        action_layout = QVBoxLayout()
        self.btn_import = QPushButton("Importer CSV")
        self.btn_import.clicked.connect(self.import_csv)
        self.btn_export = QPushButton("Exporter PNG")
        self.btn_export.clicked.connect(canvas.export_png)
        self.btn_rotate = QPushButton("Rotation auto")
        self.btn_rotate.clicked.connect(self.toggle_rotation)
        self.btn_reset = QPushButton("Réinitialiser vue")
        self.btn_reset.clicked.connect(self.reset_view)
        action_layout.addWidget(self.btn_import)
        action_layout.addWidget(self.btn_export)
        action_layout.addWidget(self.btn_rotate)
        action_layout.addWidget(self.btn_reset)
        action_layout.addWidget(QLabel("Échelle de couleur"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "cividis", "coolwarm"])
        self.cmap_combo.currentTextChanged.connect(canvas.update_colormap)
        action_layout.addWidget(self.cmap_combo)
        action_layout.addStretch()
        main_layout.addLayout(action_layout, 1)

        # Style uniforme
        for btn in self.findChildren(QPushButton):
            btn.setMinimumHeight(35)
            btn.setStyleSheet("font-size: 11pt;")

    # Méthodes
    def toggle_rotation(self):
        self.rotating = not self.rotating
        if self.rotating:
            self._rotate_step()

    def _rotate_step(self):
        if not self.rotating:
            return
        self.canvas.rotate(d_azim=self.rotation_speed)
        QTimer.singleShot(150, self._rotate_step)  # rotation lente

    def reset_view(self):
        self.rotating = False
        self.canvas.reset_view()

    def import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(
            None, "Choisir un fichier CSV", "", "CSV (*.csv)"
        )
        if fname:
            try:
                model = RoadDataModel(fname)
                self.canvas.update_surface(model.Z)
                self.canvas.reset_view()
                QMessageBox.information(None, "Succès", f"Fichier chargé : {fname}")
            except Exception as e:
                QMessageBox.critical(None, "Erreur CSV", f"Impossible de lire '{fname}':\n{e}")

# ============================================================
#                  APPLICATION PRINCIPALE
# ============================================================

class RoadQualityApp(QWidget):
    """Fenêtre principale"""
    def __init__(self, csv_file: str = None):
        super().__init__()
        self.setWindowTitle("Surveillance Qualité Route – 3D")
        self.resize(1000, 650)

        initial_Z = np.zeros((8,8)) if csv_file is None else RoadDataModel(csv_file).Z

        canvas = Surface3DCanvas(initial_Z)
        controls = ControlPanel(canvas)

        layout = QVBoxLayout(self)
        layout.addWidget(canvas, stretch=1)
        layout.addWidget(controls)

# ============================================================
#                          MAIN
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RoadQualityApp()  # Aucun CSV initial obligatoire
    window.show()
    sys.exit(app.exec())
