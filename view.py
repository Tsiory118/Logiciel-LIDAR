import sys
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


# =========================================================
# ====================== MODEL ============================
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
        except:
            return np.zeros((5, 5))


# =========================================================
# ====================== ANALYTICS ========================
# =========================================================

class RoadAnalytics:

    @staticmethod
    def compute(Z):
        Z = Z[~np.isnan(Z)]
        avg = np.mean(Z) / 10
        maxv = np.max(Z) / 10
        std = np.std(Z) / 10

        if maxv < 1:
            state = "Bonne"
            color = "#00e676"
        elif maxv < 3:
            state = "Moyenne"
            color = "#ff9800"
        else:
            state = "Critique"
            color = "#ff1744"

        return avg, maxv, std, state, color


# =========================================================
# ====================== 3D VIEW ==========================
# =========================================================

class Surface3D(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(facecolor="#121212")
        super().__init__(fig)

        self.ax = fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#121212")

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
        self.update_surface(self.Z)

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
# ====================== DASHBOARD ========================
# =========================================================

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()

        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.rotate_surface)
        self.rotating = False

        main_layout = QHBoxLayout(self)

        sidebar = QVBoxLayout()
        sidebar.setAlignment(Qt.AlignTop)

        self.btn_import = QPushButton("Importer CSV")
        self.btn_rotate = QPushButton("Démarrer Rotation")
        self.btn_default = QPushButton("Vue Isométrique")
        self.btn_top = QPushButton("Vue Haut")
        self.btn_profile = QPushButton("Vue Profil")
        self.btn_report = QPushButton("Générer Rapport")

        for btn in [
            self.btn_import,
            self.btn_rotate,
            self.btn_default,
            self.btn_top,
            self.btn_profile,
            self.btn_report
        ]:
            btn.setCursor(Qt.PointingHandCursor)
            sidebar.addWidget(btn)

        self.analytics_label = QLabel("Aucune donnée chargée")
        self.analytics_label.setStyleSheet("padding:10px; font-size:13px;")
        sidebar.addWidget(self.analytics_label)

        sidebar_frame = QFrame()
        sidebar_frame.setLayout(sidebar)
        sidebar_frame.setFixedWidth(260)

        self.surface3d = Surface3D()

        main_layout.addWidget(sidebar_frame)
        main_layout.addWidget(self.surface3d)

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

        avg, maxv, std, state, color = RoadAnalytics.compute(Z)

        self.analytics_label.setText(
            f"""
            <b>Analyse</b><br>
            Moyenne: {avg:.2f} cm<br>
            Max: {maxv:.2f} cm<br>
            Écart-type: {std:.2f} cm<br>
            État: <span style='color:{color}'>{state}</span>
            """
        )

    def toggle_rotation(self):
        if not self.rotating:
            self.rotation_timer.start(30)
            self.btn_rotate.setText("Arrêter Rotation")
            self.rotating = True
        else:
            self.rotation_timer.stop()
            self.btn_rotate.setText("Démarrer Rotation")
            self.rotating = False

    def rotate_surface(self):
        self.surface3d.rotate_step()

    def export_report(self):
        QMessageBox.information(self, "Rapport",
                                "Module rapport prêt pour version commerciale.")


# =========================================================
# ====================== SPLASH PREMIUM ===================
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
# ====================== DARK STYLE =======================
# =========================================================

DARK_STYLE = """
QWidget { background:#121212; color:white; font-family:Segoe UI; }

QPushButton {
    background:#1e1e1e;
    border:1px solid #333;
    padding:10px;
    border-radius:8px;
    font-size:14px;
}

QPushButton:hover { background:#2c2c2c; }
QPushButton:pressed { background:#3700b3; }

QFrame {
    background:#181818;
    border-radius:12px;
}
"""


# =========================================================
# ====================== MAIN =============================
# =========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)

    splash = SplashScreen()
    splash.show()

    sys.exit(app.exec())
