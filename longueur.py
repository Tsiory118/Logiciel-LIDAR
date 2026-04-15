import sys
import io
import re
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame,
    QMessageBox, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

# =========================================================
# MODEL
# =========================================================

class RoadDataModel:
    def __init__(self, path):
        self.path = path
        self.Z = self.load()

    def load(self):
        try:
            data = np.genfromtxt(self.path, delimiter=",", skip_header=1)

            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)

            return data[:, 3:]
        except Exception as e:
            print("Erreur CSV :", e)
            return np.zeros((8, 16))

# =========================================================
# ANALYTICS
# =========================================================

class RoadAnalytics:
    @staticmethod
    def compute(Z):
        Z_clean = Z[~np.isnan(Z)]

        avg = np.mean(Z_clean) / 10
        maxv = np.max(Z_clean) / 10
        std = np.std(Z_clean) / 10

        if maxv < 1:
            state = "Bonne"
            color = "#00e676"
            interpretation = "Surface homogène et conforme."
        elif maxv < 3:
            state = "Moyenne"
            color = "#ff9800"
            interpretation = "Irrégularités modérées."
        else:
            state = "Critique"
            color = "#ff1744"
            interpretation = "Défauts critiques."

        return avg, maxv, std, state, color, interpretation

# =========================================================
# 3D VIEW
# =========================================================

class Surface3D(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(facecolor="#0f0f0f")
        super().__init__(fig)
        self.ax = fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#0f0f0f")
        self.elev = 30
        self.azim = -60
        self.Z = np.zeros((5, 5))

    def update_surface(self, Z):
        self.Z = Z
        self.ax.clear()

        rows, cols = Z.shape
        X, Y = np.meshgrid(range(cols), range(rows))

        self.ax.plot_surface(X, Y, Z, cmap="inferno", edgecolor="none")

        self.ax.view_init(self.elev, self.azim)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        self.draw_idle()

    def rotate_step(self):
        self.azim += 1
        self.ax.view_init(self.elev, self.azim)
        self.draw_idle()

# =========================================================
# DASHBOARD
# =========================================================

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RouBot Application")
        self.resize(1500, 850)

        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.rotate_surface)
        self.rotating = False

        main_layout = QHBoxLayout(self)

        # SIDEBAR
        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar_frame)

        self.btn_import = QPushButton("📂 Importer CSV")
        self.btn_rotate = QPushButton("🔄 Rotation")
        self.btn_report = QPushButton("📄 Rapport PDF")

        sidebar_layout.addWidget(self.btn_import)
        sidebar_layout.addWidget(self.btn_rotate)
        sidebar_layout.addWidget(self.btn_report)

        self.analytics_label = QLabel("Aucune donnée")
        self.analytics_label.setWordWrap(True)
        sidebar_layout.addWidget(self.analytics_label)

        # 3D VIEW
        self.surface3d = Surface3D()

        main_layout.addWidget(sidebar_frame)
        main_layout.addWidget(self.surface3d)

        # EVENTS
        self.btn_import.clicked.connect(self.import_csv)
        self.btn_rotate.clicked.connect(self.toggle_rotation)
        self.btn_report.clicked.connect(self.export_report)

    # ✅ VERSION CORRIGÉE
    def import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, "CSV", "", "CSV (*.csv)")
        if not fname:
            return

        model = RoadDataModel(fname)
        Z = model.Z
        self.surface3d.update_surface(Z)

        avg, maxv, std, state, color, interpretation = RoadAnalytics.compute(Z)

        # LONGUEUR ROUTE
        nb_lignes = Z.shape[0]
        longueur_m = (nb_lignes / 8 * 25) / 100

        self.analytics_label.setText(f"""
        <b>Analyse de surface routière</b><br>
        - Longueur: {longueur_m:.2f} m<br>
        - Moyenne: {avg:.2f} cm<br>
        - Max: {maxv:.2f} cm<br>
        - Écart-type: {std:.2f} cm<br><br>

        <b>État:</b> {state}<br>
        {interpretation}
        """)

    def toggle_rotation(self):
        if self.rotating:
            self.rotation_timer.stop()
        else:
            self.rotation_timer.start(30)

        self.rotating = not self.rotating

    def rotate_surface(self):
        self.surface3d.rotate_step()

    def export_report(self):
        fname, _ = QFileDialog.getSaveFileName(self, "PDF", "", "*.pdf")
        if not fname:
            return

        buf = io.BytesIO()
        self.surface3d.figure.savefig(buf, format='png')
        buf.seek(0)

        doc = SimpleDocTemplate(fname, pagesize=A4)
        elements = []

        styles = getSampleStyleSheet()

        elements.append(Paragraph("Analyse Routière", styles['Title']))
        elements.append(Spacer(1, 10))

        img = Image(buf)
        img.drawHeight = 300
        img.drawWidth = 400
        elements.append(img)

        txt = re.sub(r"<.*?>", "", self.analytics_label.text())
        elements.append(Paragraph(txt, styles['Normal']))

        doc.build(elements)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec())