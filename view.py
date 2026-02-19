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
# ====================== MODEL ===========================
# =========================================================

class RoadDataModel:
    def __init__(self, path):
        self.path = path
        self.Z = self.load()

    def load(self):
        try:
            data = np.genfromtxt(self.path, delimiter=",")
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            return data[:, 1:]
        except Exception:
            return np.zeros((5, 5))

# =========================================================
# ====================== ANALYTICS =======================
# =========================================================

class RoadAnalytics:
    @staticmethod
    def compute(Z):
        Z_clean = Z[~np.isnan(Z)]
        avg = np.mean(Z_clean) / 10  # cm
        maxv = np.max(Z_clean) / 10
        std = np.std(Z_clean) / 10

        if maxv < 1:
            state = "Bonne"
            color = "#00e676"
            interpretation = "Surface homogène et conforme, peu d'irrégularités."
        elif maxv < 3:
            state = "Moyenne"
            color = "#ff9800"
            interpretation = (
                "Irrégularités modérées détectées. "
                "Surveillance et entretien ponctuel recommandé."
            )
        else:
            state = "Critique"
            color = "#ff1744"
            interpretation = (
                "Défauts critiques présents. "
                "Planifier travaux de nivellement ou réparation ciblée."
            )
        return avg, maxv, std, state, color, interpretation

# =========================================================
# ====================== 3D VIEW =========================
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
        self.ax.plot_surface(
            X, Y, Z,
            cmap="inferno",
            edgecolor="none",
            antialiased=True,
            shade=True
        )
        self.ax.view_init(self.elev, self.azim)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.draw_idle()

    def rotate_step(self):
        self.azim += 1
        self.ax.view_init(self.elev, self.azim)
        self.draw_idle()

    def view_top(self):
        self.elev = 90
        self.azim = -90
        self.update_surface(self.Z)

    def view_profile(self):
        self.elev = 0
        self.azim = -90
        self.update_surface(self.Z)

    def view_default(self):
        self.elev = 30
        self.azim = -60
        self.update_surface(self.Z)

# =========================================================
# ====================== DASHBOARD =======================
# =========================================================

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()

        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.rotate_surface)
        self.rotating = False

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ================= SIDEBAR =================
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("Sidebar")
        sidebar_frame.setFixedWidth(280)

        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(15)

        title = QLabel("K-Route Dashboard")
        title.setStyleSheet("font-size:18px; font-weight:600; color:#00e5ff;")
        sidebar_layout.addWidget(title)
        sidebar_layout.addSpacing(10)

        self.btn_import = QPushButton("📂  Importer CSV")
        self.btn_rotate = QPushButton("🔄  Rotation Auto")
        self.btn_default = QPushButton("🧭  Vue Isométrique")
        self.btn_top = QPushButton("⬆  Vue Haut")
        self.btn_profile = QPushButton("➡  Vue Profil")
        self.btn_report = QPushButton("📄  Générer Rapport")

        for btn in [
            self.btn_import,
            self.btn_rotate,
            self.btn_default,
            self.btn_top,
            self.btn_profile,
            self.btn_report
        ]:
            btn.setCursor(Qt.PointingHandCursor)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        # ================= ANALYTICS CARD =================
        self.analytics_card = QFrame()
        self.analytics_card.setObjectName("Card")

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setOffset(0, 0)
        shadow.setColor(QColor(0, 229, 255, 60))
        self.analytics_card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(self.analytics_card)
        card_layout.setContentsMargins(15, 15, 15, 15)

        analytics_title = QLabel("Analyse Surface")
        analytics_title.setStyleSheet("font-size:16px; font-weight:bold; color:#00e5ff;")

        self.analytics_label = QLabel("Aucune donnée chargée")
        self.analytics_label.setWordWrap(True)
        self.analytics_label.setStyleSheet("font-size:13px; color:#cccccc;")

        card_layout.addWidget(analytics_title)
        card_layout.addSpacing(10)
        card_layout.addWidget(self.analytics_label)

        sidebar_layout.addWidget(self.analytics_card)

        # ================= 3D VIEW =================
        self.surface3d = Surface3D()

        main_layout.addWidget(sidebar_frame)
        main_layout.addWidget(self.surface3d)

        # ================= CONNECTIONS =================
        self.btn_import.clicked.connect(self.import_csv)
        self.btn_rotate.clicked.connect(self.toggle_rotation)
        self.btn_top.clicked.connect(self.surface3d.view_top)
        self.btn_profile.clicked.connect(self.surface3d.view_profile)
        self.btn_default.clicked.connect(self.surface3d.view_default)
        self.btn_report.clicked.connect(self.export_report)

    def import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Importer CSV", "", "CSV (*.csv)")
        if not fname:
            return

        model = RoadDataModel(fname)
        Z = model.Z
        self.surface3d.update_surface(Z)

        avg, maxv, std, state, color, interpretation = RoadAnalytics.compute(Z)

        self.analytics_label.setText(
            f"""
            <b>Analyse de surface routière</b><br>
            - Hauteur moyenne des irrégularités: {avg:.2f} cm<br>
            - Déviation maximale détectée: {maxv:.2f} cm<br>
            - Écart-type (uniformité): {std:.2f} cm<br>
            - État général: <span style='color:{color}'>{state}</span><br>
            <br>
            <b>Interprétation :</b><br>
            {interpretation}
            """
        )

    def toggle_rotation(self):
        if not self.rotating:
            self.rotation_timer.start(16)
            self.btn_rotate.setText("⏹  Stop Rotation")
            self.rotating = True
        else:
            self.rotation_timer.stop()
            self.btn_rotate.setText("🔄  Rotation Auto")
            self.rotating = False

    def rotate_surface(self):
        self.surface3d.rotate_step()

    def export_report(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer Rapport PDF", "", "PDF (*.pdf)"
        )
        if not fname:
            return
        if not fname.endswith(".pdf"):
            fname += ".pdf"

        try:
            # Capture image 3D
            buf = io.BytesIO()
            self.surface3d.figure.savefig(buf, format='png', facecolor=self.surface3d.figure.get_facecolor())
            buf.seek(0)

            # Créer le PDF
            doc = SimpleDocTemplate(fname, pagesize=A4,
                                    rightMargin=2*cm, leftMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            elements = []

            # Styles
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='AnalysisText',
                fontName='Helvetica',
                fontSize=12,
                leading=16,
                textColor=colors.white,
                backColor=colors.HexColor('#0f0f0f')
            ))
            styles.add(ParagraphStyle(
                name='MyTitle',
                fontName='Helvetica-Bold',
                fontSize=16,
                leading=20,
                textColor=colors.cyan,
                backColor=colors.HexColor('#0f0f0f')
            ))

            # Titre
            elements.append(Paragraph("Analyse de Surface Routière", styles['MyTitle']))
            elements.append(Spacer(1, 0.5*cm))

            # Image 3D
            img = Image(buf)
            img.drawHeight = 12*cm
            img.drawWidth = 18*cm
            elements.append(img)
            elements.append(Spacer(1, 0.5*cm))

            # Texte complet
            analysis_text = self.analytics_label.text()
            analysis_text = re.sub(r"<br\s*/?>", "\n", analysis_text)
            analysis_text = re.sub(r"<.*?>", "", analysis_text)
            elements.append(Paragraph(analysis_text, styles['AnalysisText']))

            # Générer PDF
            doc.build(elements)

            QMessageBox.information(self, "Rapport", f"Rapport enregistré avec succès:\n{fname}")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de générer le PDF:\n{str(e)}")

# =========================================================
# ====================== SPLASH ==========================
# =========================================================

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(650, 380)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.container = QFrame(self)
        self.container.setGeometry(0, 0, 650, 380)
        self.container.setStyleSheet("""
            background-color: #121212;
            border-radius: 20px;
        """)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 0)
        shadow.setColor(QColor(0, 229, 255, 120))
        self.container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self.container)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("K-Route Application")
        title.setStyleSheet("font-size:32px; font-weight:700; color:#00e5ff;")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Surveiller la qualité de la route")
        subtitle.setStyleSheet("font-size:14px; color:#888;")
        subtitle.setAlignment(Qt.AlignCenter)

        self.progress_bg = QFrame()
        self.progress_bg.setFixedSize(400, 8)
        self.progress_bg.setStyleSheet("background:#2a2a2a; border-radius:4px;")

        self.progress_bar = QFrame(self.progress_bg)
        self.progress_bar.setFixedSize(0, 8)
        self.progress_bar.setStyleSheet("""
            background:qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #00e5ff,
                stop:1 #3700b3
            );
            border-radius:4px;
        """)

        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(15)
        layout.addWidget(subtitle)
        layout.addSpacing(60)
        layout.addWidget(self.progress_bg, alignment=Qt.AlignCenter)
        layout.addStretch()

        # Fade in
        self.setWindowOpacity(0)
        self.fade_in = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in.setDuration(800)
        self.fade_in.setStartValue(0)
        self.fade_in.setEndValue(1)
        self.fade_in.start()

        # Progress animation
        self.anim = QPropertyAnimation(self.progress_bar, b"minimumWidth")
        self.anim.setDuration(3000)
        self.anim.setStartValue(0)
        self.anim.setEndValue(400)
        self.anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.anim.finished.connect(self.finish)
        self.anim.start()

    def finish(self):
        self.fade_out = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out.setDuration(500)
        self.fade_out.setStartValue(1)
        self.fade_out.setEndValue(0)
        self.fade_out.finished.connect(self.launch_main)
        self.fade_out.start()

    def launch_main(self):
        self.close()
        self.main = Dashboard()
        self.main.setWindowTitle("K-Route Application <TBag & Meik>")
        self.main.resize(1500, 850)
        self.main.show()

# =========================================================
# ====================== DARK STYLE ======================
# =========================================================

DARK_STYLE = """
QWidget {
    background-color: #0f0f0f;
    color: white;
    font-family: Segoe UI;
}

#Sidebar {
    background-color: #141414;
    border-right: 1px solid #2a2a2a;
}

#Card {
    background-color: #1c1c1c;
    border-radius: 16px;
    border: 1px solid #2a2a2a;
}

QPushButton {
    background-color: #1e1e1e;
    border: 1px solid #2f2f2f;
    padding: 12px;
    border-radius: 12px;
    font-size: 14px;
    text-align: left;
}

QPushButton:hover {
    background-color: #252525;
    border: 1px solid #00e5ff;
}

QPushButton:pressed {
    background-color: #3700b3;
}
"""

# =========================================================
# ====================== MAIN ============================
# =========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)

    splash = SplashScreen()
    splash.show()

    sys.exit(app.exec())
