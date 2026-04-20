"""
RouBot Application — Road Surface Quality Analyzer
Mémoire d'ingénieur en Informatique

Architecture : MVC (Model-View-Controller)
Thème        : Light Mode — Professionnel
"""

import sys
import io
import os
import tempfile
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame,
    QGraphicsDropShadowEffect, QSizePolicy, QScrollArea,
    QProgressBar, QStatusBar, QMainWindow, QToolBar,
    QSplitter, QGridLayout, QGroupBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSpacerItem
)
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve,
    QSequentialAnimationGroup, QParallelAnimationGroup, QRect
)
from PySide6.QtGui import (
    QColor, QFont, QPixmap, QPalette, QIcon, QLinearGradient,
    QPainter, QBrush, QPen, QFontDatabase
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage, KeepTogether
)
from reportlab.pdfgen import canvas as rl_canvas


# =========================================================
# ====================== CONSTANTS =======================
# =========================================================

APP_NAME    = "RouBot Analyzer"
APP_VERSION = "1.0.0"
APP_AUTHOR  = "TBag & Meik"

PALETTE = {
    "bg_primary"   : "#F8FAFC",
    "bg_secondary" : "#FFFFFF",
    "bg_card"      : "#FFFFFF",
    "sidebar_bg"   : "#1E293B",
    "sidebar_text" : "#94A3B8",
    "sidebar_hover": "#334155",
    "accent_blue"  : "#2563EB",
    "accent_cyan"  : "#0EA5E9",
    "accent_green" : "#10B981",
    "accent_orange": "#F59E0B",
    "accent_red"   : "#EF4444",
    "text_primary" : "#0F172A",
    "text_secondary": "#475569",
    "text_muted"   : "#94A3B8",
    "border"       : "#E2E8F0",
    "border_focus" : "#2563EB",
    "shadow"       : "rgba(15,23,42,0.08)",
}

APP_STYLE = """
/* ── Global ──────────────────────────────────────── */
QApplication, QMainWindow {
    background-color: #F8FAFC;
}

QWidget {
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 13px;
    color: #0F172A;
}

QMainWindow {
    background-color: #F8FAFC;
}

/* ── Scrollbar ───────────────────────────────────── */
QScrollBar:vertical {
    background: #F1F5F9;
    width: 6px;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #CBD5E1;
    border-radius: 3px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #94A3B8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

/* ── Sidebar ─────────────────────────────────────── */
#Sidebar {
    background-color: #1E293B;
    border-right: 1px solid #334155;
}

/* ── Cards ───────────────────────────────────────── */
#MetricCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
}

#SectionCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
}

/* ── Nav Buttons ─────────────────────────────────── */
#NavBtn {
    background-color: transparent;
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    text-align: left;
    color: #94A3B8;
    font-size: 13px;
    font-weight: 500;
}
#NavBtn:hover {
    background-color: #334155;
    color: #F1F5F9;
}
#NavBtn[active="true"] {
    background-color: #2563EB;
    color: #FFFFFF;
}

/* ── Primary Button ──────────────────────────────── */
#PrimaryBtn {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 600;
}
#PrimaryBtn:hover  { background-color: #1D4ED8; }
#PrimaryBtn:pressed { background-color: #1E40AF; }

/* ── Secondary Button ────────────────────────────── */
#SecondaryBtn {
    background-color: #F1F5F9;
    color: #0F172A;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 500;
}
#SecondaryBtn:hover  { background-color: #E2E8F0; border-color: #CBD5E1; }
#SecondaryBtn:pressed { background-color: #CBD5E1; }

/* ── Status Badge ────────────────────────────────── */
#BadgeGood    { background:#DCFCE7; color:#166534; border-radius:6px; padding:2px 10px; font-weight:600; }
#BadgeWarning { background:#FEF3C7; color:#92400E; border-radius:6px; padding:2px 10px; font-weight:600; }
#BadgeCritical{ background:#FEE2E2; color:#991B1B; border-radius:6px; padding:2px 10px; font-weight:600; }

/* ── Tab Widget ──────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #E2E8F0;
    border-radius: 0 12px 12px 12px;
    background: #FFFFFF;
}
QTabBar::tab {
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-bottom: none;
    padding: 8px 20px;
    border-radius: 8px 8px 0 0;
    color: #64748B;
    font-weight: 500;
}
QTabBar::tab:selected { background: #FFFFFF; color: #2563EB; font-weight: 600; }
QTabBar::tab:hover:!selected { background: #E2E8F0; }

/* ── Table ───────────────────────────────────────── */
QTableWidget {
    background: #FFFFFF;
    gridline-color: #E2E8F0;
    border: none;
    border-radius: 8px;
}
QTableWidget::item { padding: 8px 12px; }
QTableWidget::item:selected { background: #EFF6FF; color: #1E40AF; }
QHeaderView::section {
    background: #F8FAFC;
    border: none;
    border-bottom: 2px solid #E2E8F0;
    padding: 10px 12px;
    font-weight: 600;
    color: #475569;
}

/* ── Progress Bar ────────────────────────────────── */
QProgressBar {
    background: #E2E8F0;
    border-radius: 6px;
    height: 8px;
    text-align: center;
    font-size: 11px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2563EB, stop:1 #0EA5E9);
    border-radius: 6px;
}

/* ── Status Bar ──────────────────────────────────── */
QStatusBar {
    background: #FFFFFF;
    border-top: 1px solid #E2E8F0;
    color: #64748B;
    font-size: 12px;
}
"""


# =========================================================
# ====================== MODEL ===========================
# =========================================================

class RoadDataModel:
    """Charge et normalise les données CSV du capteur routier."""

    def __init__(self, path: str):
        self.path      = path
        self.filename  = path.split("/")[-1].split("\\")[-1]
        self.load_time = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        self.Z, self.raw_rows, self.raw_cols = self._load()

    def _load(self):
        try:
            data = np.genfromtxt(self.path, delimiter=",", skip_header=1)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            Z = data[:, 3:].astype(float)
            return Z, data.shape[0], Z.shape[1]
        except Exception as e:
            print(f"[RoadDataModel] Erreur chargement : {e}")
            return np.zeros((8, 16)), 0, 0


# =========================================================
# ====================== ANALYTICS =======================
# =========================================================

class RoadAnalytics:
    """Calcule tous les indicateurs de qualité de surface."""

    SEUIL_BON      = 1.0
    SEUIL_MOYEN    = 3.0
    SAMPLE_RATE    = 8
    SENSOR_SPACING = 0.125

    @classmethod
    def compute(cls, Z: np.ndarray) -> dict:
        Z_clean = Z[~np.isnan(Z)]
        if Z_clean.size == 0:
            return {}

        avg    = float(np.mean(Z_clean)) / 10
        maxv   = float(np.max(Z_clean))  / 10
        minv   = float(np.min(Z_clean))  / 10
        std    = float(np.std(Z_clean))  / 10
        median = float(np.median(Z_clean)) / 10

        rugosite  = float(np.std(Z))
        variation = float(np.max(Z) - np.min(Z))

        threshold    = np.mean(Z) + 2 * np.std(Z)
        nb_obstacles = int(np.count_nonzero(Z > threshold))

        profil_long = np.mean(Z, axis=1)
        if len(profil_long) > 1:
            coeffs = np.polyfit(range(len(profil_long)), profil_long, 1)
            pente  = float(coeffs[0])
        else:
            pente = 0.0

        nb_lignes  = Z.shape[0]
        longueur_m = float((nb_lignes / 8) * 15) / 100

        profil_transv = np.mean(Z, axis=0)

        if maxv < cls.SEUIL_BON:
            state          = "Bonne"
            badge          = "Good"
            color          = "#10B981"
            interpretation = "Surface homogène, conforme aux normes routières."
            iri_score      = "IRI < 1  —  Très bon état"
        elif maxv < cls.SEUIL_MOYEN:
            state          = "Moyenne"
            badge          = "Warning"
            color          = "#F59E0B"
            interpretation = "Irrégularités modérées, surveillance recommandée."
            iri_score      = "IRI 1–3  —  État acceptable"
        else:
            state          = "Critique"
            badge          = "Critical"
            color          = "#EF4444"
            interpretation = "Défauts critiques, intervention urgente requise."
            iri_score      = "IRI > 3  —  Mauvais état"

        quality_score = max(0, min(100, int(100 - (maxv / 6) * 100)))

        return {
            "avg": avg, "maxv": maxv, "minv": minv, "std": std, "median": median,
            "rugosite": rugosite, "variation": variation,
            "nb_obstacles": nb_obstacles, "pente": pente,
            "longueur_m": longueur_m,
            "profil_long": profil_long, "profil_transv": profil_transv,
            "state": state, "badge": badge, "color": color,
            "interpretation": interpretation, "iri_score": iri_score,
            "quality_score": quality_score,
            "Z": Z,
        }


# =========================================================
# ====================== PDF GENERATOR ===================
# =========================================================

def _hex_to_rl_color(hex_color: str):
    """Convertit une couleur hex en objet ReportLab Color."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return colors.Color(r, g, b)


class PDFReportGenerator:
    """Génère un rapport PDF professionnel avec ReportLab."""

    # Couleurs du thème
    C_BLUE   = _hex_to_rl_color("#2563EB")
    C_CYAN   = _hex_to_rl_color("#0EA5E9")
    C_GREEN  = _hex_to_rl_color("#10B981")
    C_ORANGE = _hex_to_rl_color("#F59E0B")
    C_RED    = _hex_to_rl_color("#EF4444")
    C_DARK   = _hex_to_rl_color("#1E293B")
    C_GRAY   = _hex_to_rl_color("#64748B")
    C_LIGHT  = _hex_to_rl_color("#F8FAFC")
    C_BORDER = _hex_to_rl_color("#E2E8F0")
    C_WHITE  = colors.white

    def __init__(self, output_path: str, model: "RoadDataModel", stats: dict):
        self.output_path = output_path
        self.model       = model
        self.stats       = stats
        self.styles      = self._build_styles()
        self._tmp_files  = []  # fichiers temporaires à supprimer

    def _build_styles(self):
        base = getSampleStyleSheet()
        custom = {
            "Title": ParagraphStyle(
                "Title", parent=base["Normal"],
                fontSize=26, textColor=self.C_DARK,
                fontName="Helvetica-Bold",
                spaceAfter=4, alignment=TA_LEFT,
            ),
            "Subtitle": ParagraphStyle(
                "Subtitle", parent=base["Normal"],
                fontSize=12, textColor=self.C_GRAY,
                fontName="Helvetica",
                spaceAfter=2, alignment=TA_LEFT,
            ),
            "SectionHeader": ParagraphStyle(
                "SectionHeader", parent=base["Normal"],
                fontSize=13, textColor=self.C_BLUE,
                fontName="Helvetica-Bold",
                spaceBefore=14, spaceAfter=6,
                borderPadding=(0, 0, 4, 0),
            ),
            "Body": ParagraphStyle(
                "Body", parent=base["Normal"],
                fontSize=10, textColor=self.C_DARK,
                fontName="Helvetica",
                spaceAfter=6, leading=15,
            ),
            "BodyBold": ParagraphStyle(
                "BodyBold", parent=base["Normal"],
                fontSize=10, textColor=self.C_DARK,
                fontName="Helvetica-Bold",
                spaceAfter=4,
            ),
            "Footer": ParagraphStyle(
                "Footer", parent=base["Normal"],
                fontSize=8, textColor=self.C_GRAY,
                fontName="Helvetica",
                alignment=TA_CENTER,
            ),
            "Caption": ParagraphStyle(
                "Caption", parent=base["Normal"],
                fontSize=8, textColor=self.C_GRAY,
                fontName="Helvetica-Oblique",
                alignment=TA_CENTER, spaceAfter=8,
            ),
            "StateGood": ParagraphStyle(
                "StateGood", parent=base["Normal"],
                fontSize=11, textColor=self.C_GREEN,
                fontName="Helvetica-Bold",
            ),
            "StateWarning": ParagraphStyle(
                "StateWarning", parent=base["Normal"],
                fontSize=11, textColor=self.C_ORANGE,
                fontName="Helvetica-Bold",
            ),
            "StateCritical": ParagraphStyle(
                "StateCritical", parent=base["Normal"],
                fontSize=11, textColor=self.C_RED,
                fontName="Helvetica-Bold",
            ),
        }
        return custom

    # ── Génération des figures matplotlib ──────────────────

    def _render_fig_to_tmp(self, fig, suffix="_chart.png", dpi=150) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        tmp.close()
        self._tmp_files.append(tmp.name)
        return tmp.name

    def _make_profiles_figure(self) -> str:
        Z             = self.stats["Z"]
        profil_long   = self.stats["profil_long"]
        profil_transv = self.stats["profil_transv"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), facecolor="white")
        BLUE  = "#2563EB"
        GREEN = "#10B981"
        CYAN  = "#0EA5E9"

        # Profil longitudinal
        ax1 = axes[0]
        x1  = np.arange(len(profil_long))
        ax1.fill_between(x1, profil_long, alpha=0.15, color=BLUE)
        ax1.plot(x1, profil_long, color=BLUE, linewidth=2, marker="o", markersize=3)
        ax1.axhline(np.mean(profil_long), color=CYAN, linestyle="--",
                    linewidth=1.2, label="Moyenne")
        ax1.set_title("Profil Longitudinal", fontsize=10, fontweight="bold",
                      color="#1E293B")
        ax1.set_xlabel("Points de mesure", fontsize=8, color="#64748B")
        ax1.set_ylabel("Hauteur (mm)", fontsize=8, color="#64748B")
        ax1.legend(fontsize=8)
        ax1.set_facecolor("#F8FAFC")
        ax1.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        for sp in ax1.spines.values(): sp.set_color("#E2E8F0")
        ax1.tick_params(colors="#64748B", labelsize=8)

        # Profil transversal
        ax2 = axes[1]
        x2  = np.arange(len(profil_transv))
        ax2.fill_between(x2, profil_transv, alpha=0.15, color=GREEN)
        ax2.plot(x2, profil_transv, color=GREEN, linewidth=2, marker="s", markersize=3)
        ax2.axhline(np.mean(profil_transv), color=CYAN, linestyle="--",
                    linewidth=1.2, label="Moyenne")
        ax2.set_title("Profil Transversal", fontsize=10, fontweight="bold",
                      color="#1E293B")
        ax2.set_xlabel("Capteurs (largeur)", fontsize=8, color="#64748B")
        ax2.set_ylabel("Hauteur (mm)", fontsize=8, color="#64748B")
        ax2.legend(fontsize=8)
        ax2.set_facecolor("#F8FAFC")
        ax2.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        for sp in ax2.spines.values(): sp.set_color("#E2E8F0")
        ax2.tick_params(colors="#64748B", labelsize=8)

        fig.tight_layout()
        path = self._render_fig_to_tmp(fig, "_profiles.png")
        plt.close(fig)
        return path

    def _make_heatmap_hist_figure(self) -> str:
        Z = self.stats["Z"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), facecolor="white")
        BLUE = "#2563EB"
        CYAN = "#0EA5E9"
        RED  = "#EF4444"

        # Heatmap
        ax3 = axes[0]
        im  = ax3.imshow(Z, cmap="RdYlGn_r", aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax3, shrink=0.9, label="mm", pad=0.02)
        ax3.set_title("Carte Thermique des Défauts", fontsize=10, fontweight="bold",
                      color="#1E293B")
        ax3.set_xlabel("Capteurs (largeur)", fontsize=8, color="#64748B")
        ax3.set_ylabel("Points de mesure", fontsize=8, color="#64748B")
        ax3.tick_params(colors="#64748B", labelsize=8)

        # Histogramme
        ax4    = axes[1]
        Z_flat = Z.flatten()
        threshold = np.mean(Z_flat) + 2 * np.std(Z_flat)
        n, bins, patches = ax4.hist(Z_flat, bins=20, color=BLUE,
                                     edgecolor="white", linewidth=0.5, alpha=0.85)
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(RED if left > threshold else BLUE)
        ax4.axvline(np.mean(Z_flat), color=CYAN, linestyle="--",
                    linewidth=1.5, label="Moyenne")
        ax4.axvline(threshold, color=RED, linestyle=":",
                    linewidth=1.5, label="Seuil obstacles")
        ax4.legend(fontsize=8)
        ax4.set_title("Distribution des Hauteurs", fontsize=10, fontweight="bold",
                      color="#1E293B")
        ax4.set_xlabel("Hauteur (mm)", fontsize=8, color="#64748B")
        ax4.set_ylabel("Frequence", fontsize=8, color="#64748B")
        ax4.set_facecolor("#F8FAFC")
        ax4.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        for sp in ax4.spines.values(): sp.set_color("#E2E8F0")
        ax4.tick_params(colors="#64748B", labelsize=8)

        fig.tight_layout()
        path = self._render_fig_to_tmp(fig, "_heatmap_hist.png")
        plt.close(fig)
        return path

    def _make_3d_figure(self) -> str:
        Z = self.stats["Z"]
        cmap = LinearSegmentedColormap.from_list(
            "road", ["#1D4ED8", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444"]
        )
        fig = plt.figure(figsize=(8, 5), facecolor="white")
        ax  = fig.add_subplot(111, projection="3d")
        rows, cols = Z.shape
        X, Y = np.meshgrid(range(cols), range(rows))
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none",
                               antialiased=True, shade=True, alpha=0.95)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,
                     label="Hauteur (mm)", pad=0.1)
        ax.set_facecolor("#F8FAFC")
        ax.set_xlabel("Largeur (capteurs)", fontsize=8)
        ax.set_ylabel("Longueur (points)", fontsize=8)
        ax.set_zlabel("Hauteur (mm)", fontsize=8)
        ax.set_title("Modele Surfacique 3D", fontsize=11,
                     fontweight="bold", color="#1E293B")
        ax.view_init(30, -60)

        fig.tight_layout()
        path = self._render_fig_to_tmp(fig, "_3d.png", dpi=120)
        plt.close(fig)
        return path

    # ── Construction du document ────────────────────────────

    def _header_footer(self, canvas_obj, doc):
        """Dessine l'en-tête et le pied de page sur chaque page."""
        canvas_obj.saveState()
        W, H = A4

        # Bande bleue en haut
        canvas_obj.setFillColor(self.C_DARK)
        canvas_obj.rect(0, H - 48 * mm, W, 48 * mm, fill=1, stroke=0)

        # Titre dans la bande
        canvas_obj.setFont("Helvetica-Bold", 18)
        canvas_obj.setFillColor(colors.white)
        canvas_obj.drawString(20 * mm, H - 22 * mm, APP_NAME)

        canvas_obj.setFont("Helvetica", 9)
        canvas_obj.setFillColor(_hex_to_rl_color("#94A3B8"))
        canvas_obj.drawString(20 * mm, H - 31 * mm,
                              "Rapport d'Analyse de Surface Routiere")

        # Numéro de page (droite)
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(_hex_to_rl_color("#94A3B8"))
        canvas_obj.drawRightString(W - 20 * mm, H - 26 * mm,
                                   f"Page {doc.page}")

        # Trait de séparation bas
        canvas_obj.setStrokeColor(self.C_BORDER)
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(20 * mm, 14 * mm, W - 20 * mm, 14 * mm)

        # Pied de page
        canvas_obj.setFont("Helvetica", 7.5)
        canvas_obj.setFillColor(self.C_GRAY)
        canvas_obj.drawString(20 * mm, 9 * mm,
                              f"{APP_NAME} v{APP_VERSION}  |  {APP_AUTHOR}")
        canvas_obj.drawRightString(W - 20 * mm, 9 * mm,
                                   f"Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}")

        canvas_obj.restoreState()

    def _kpi_table(self) -> Table:
        s = self.stats
        data = [
            ["Indicateur", "Valeur", "Unite", "Interpretation"],
            ["Hauteur Moyenne",   f"{s['avg']:.3f}",  "cm",  "Valeur centrale de la surface"],
            ["Hauteur Maximale",  f"{s['maxv']:.3f}", "cm",  "Pic enregistre (point le + haut)"],
            ["Hauteur Minimale",  f"{s['minv']:.3f}", "cm",  "Point le plus bas"],
            ["Ecart-type",        f"{s['std']:.3f}",  "cm",  "Dispersion des mesures"],
            ["Mediane",           f"{s['median']:.3f}", "cm", "Valeur mediane"],
            ["Rugosite (sigma)",  f"{s['rugosite']:.3f}", "mm", "Rugosité brute de surface"],
            ["Variation totale",  f"{s['variation']:.3f}", "mm", "Amplitude max-min"],
            ["Nb Obstacles",      str(s['nb_obstacles']), "", "Points depassant 2 sigma"],
            ["Pente longitudinale", f"{s['pente']:.4f}", "mm/pt", "Inclinaison axiale estimee"],
            ["Longueur estimee",  f"{s['longueur_m']:.2f}", "m", "Section analysee"],
            ["Score qualite",     f"{s['quality_score']} / 100", "", s['iri_score']],
        ]

        col_widths = [5 * cm, 3 * cm, 2 * cm, 7.5 * cm]
        tbl = Table(data, colWidths=col_widths, repeatRows=1)

        style = TableStyle([
            # En-tête
            ("BACKGROUND",    (0, 0), (-1, 0), self.C_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 9),
            ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING",    (0, 0), (-1, 0), 8),
            # Corps
            ("FONTNAME",  (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",  (0, 1), (-1, -1), 9),
            ("ALIGN",     (1, 1), (2, -1), "CENTER"),
            ("ALIGN",     (0, 1), (0, -1), "LEFT"),
            ("TOPPADDING",    (0, 1), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            # Lignes alternées
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, _hex_to_rl_color("#F8FAFC")]),
            # Grille
            ("GRID",       (0, 0), (-1, -1), 0.5, self.C_BORDER),
            ("LINEBELOW",  (0, 0), (-1, 0),  1.0, self.C_BLUE),
            # Valeurs numériques en bleu
            ("TEXTCOLOR",  (1, 1), (2, -1), self.C_BLUE),
            ("FONTNAME",   (1, 1), (2, -1), "Helvetica-Bold"),
        ])
        tbl.setStyle(style)
        return tbl

    def _sensor_table(self) -> Table:
        Z = self.stats["Z"]
        threshold = np.mean(Z) + 2 * np.std(Z)

        headers = ["Capteur", "Moy (cm)", "Max (cm)", "Min (cm)",
                   "Std (cm)", "Obstacles", "Etat"]
        col_widths = [2.2*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm, 3.3*cm]

        data = [headers]
        for i, col in enumerate(Z.T):
            col_clean = col[~np.isnan(col)]
            if col_clean.size == 0:
                continue
            avg  = float(np.mean(col_clean)) / 10
            maxv = float(np.max(col_clean))  / 10
            minv = float(np.min(col_clean))  / 10
            std  = float(np.std(col_clean))  / 10
            obs  = int(np.count_nonzero(col > threshold))
            etat = "Bon" if maxv < 1 else ("Moyen" if maxv < 3 else "Critique")
            data.append([
                f"C{i+1:02d}",
                f"{avg:.3f}", f"{maxv:.3f}", f"{minv:.3f}",
                f"{std:.3f}", str(obs), etat
            ])

        tbl = Table(data, colWidths=col_widths, repeatRows=1)

        # Style de base
        style_cmds = [
            ("BACKGROUND",    (0, 0), (-1, 0), self.C_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, _hex_to_rl_color("#F8FAFC")]),
            ("GRID",    (0, 0), (-1, -1), 0.4, self.C_BORDER),
            ("LINEBELOW", (0, 0), (-1, 0), 1.0, self.C_BLUE),
        ]

        # Colorier la colonne État
        for row_idx, row in enumerate(data[1:], start=1):
            etat = row[6]
            if etat == "Bon":
                style_cmds += [
                    ("TEXTCOLOR",    (6, row_idx), (6, row_idx), self.C_GREEN),
                    ("FONTNAME",     (6, row_idx), (6, row_idx), "Helvetica-Bold"),
                ]
            elif etat == "Moyen":
                style_cmds += [
                    ("TEXTCOLOR",    (6, row_idx), (6, row_idx), self.C_ORANGE),
                    ("FONTNAME",     (6, row_idx), (6, row_idx), "Helvetica-Bold"),
                ]
            else:
                style_cmds += [
                    ("TEXTCOLOR",    (6, row_idx), (6, row_idx), self.C_RED),
                    ("FONTNAME",     (6, row_idx), (6, row_idx), "Helvetica-Bold"),
                    ("BACKGROUND",   (6, row_idx), (6, row_idx),
                     _hex_to_rl_color("#FEF2F2")),
                ]

        tbl.setStyle(TableStyle(style_cmds))
        return tbl

    def _state_badge_para(self) -> Paragraph:
        s     = self.stats
        badge = s["badge"]
        if badge == "Good":
            style_key = "StateGood"
            label     = "ETAT : BONNE"
            prefix    = "[OK] "
        elif badge == "Warning":
            style_key = "StateWarning"
            label     = "ETAT : MOYENNE"
            prefix    = "[!]  "
        else:
            style_key = "StateCritical"
            label     = "ETAT : CRITIQUE"
            prefix    = "[!!] "
        return Paragraph(f"{prefix}{label}", self.styles[style_key])

    # ── Point d'entrée ──────────────────────────────────────

    def generate(self):
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=52 * mm,   # espace pour l'en-tête
            bottomMargin=22 * mm,
            title=f"Rapport RouBot — {self.model.filename}",
            author=APP_AUTHOR,
            subject="Analyse de qualité de surface routière",
        )

        story = []
        s = self.stats

        # ── Page de garde info ──────────────────────────────
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Rapport d'Analyse par RouBot",
                                self.styles["Title"]))
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph(
            f"Fichier : {self.model.filename}   |   "
            f"Genere le {self.model.load_time}",
            self.styles["Subtitle"]
        ))
        story.append(Spacer(1, 3 * mm))
        story.append(HRFlowable(width="100%", thickness=1,
                                 color=self.C_BORDER, spaceAfter=8))

        # Résumé exécutif
        story.append(self._state_badge_para())
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(s["interpretation"], self.styles["Body"]))
        story.append(Paragraph(
            f"Score Qualite Global : <b>{s['quality_score']} / 100</b>   |   "
            f"{s['iri_score']}",
            self.styles["Body"]
        ))

        # ── Section KPI ─────────────────────────────────────
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("Indicateurs Cles de Performance (KPI)",
                                self.styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=self.C_BLUE, spaceAfter=6))
        story.append(self._kpi_table())

        # ── Section Graphiques ───────────────────────────────
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("Visualisations des Profils",
                                self.styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=self.C_BLUE, spaceAfter=6))

        path_profiles = self._make_profiles_figure()
        story.append(RLImage(path_profiles, width=17 * cm, height=6 * cm))
        story.append(Paragraph(
            "Figure 1 — Profil longitudinal (gauche) et profil transversal (droite) "
            "de la surface analysee.",
            self.styles["Caption"]
        ))

        story.append(Spacer(1, 3 * mm))
        path_hm = self._make_heatmap_hist_figure()
        story.append(RLImage(path_hm, width=17 * cm, height=6 * cm))
        story.append(Paragraph(
            "Figure 2 — Carte thermique des defauts (gauche) et distribution "
            "statistique des hauteurs (droite).",
            self.styles["Caption"]
        ))

        # ── Section 3D ───────────────────────────────────────
        story.append(PageBreak())
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("Modele Surfacique 3D",
                                self.styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=self.C_BLUE, spaceAfter=6))

        path_3d = self._make_3d_figure()
        story.append(RLImage(path_3d, width=14 * cm, height=9 * cm))
        story.append(Paragraph(
            "Figure 3 — Representation tridimensionnelle de la surface routiere. "
            "Le degradé de couleur indique les niveaux de hauteur (bleu = bas, rouge = haut).",
            self.styles["Caption"]
        ))

        # ── Section Données capteurs ─────────────────────────
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("Analyse par Capteur",
                                self.styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=self.C_BLUE, spaceAfter=6))
        story.append(Paragraph(
            f"Le tableau ci-dessous presente les statistiques individuelles de chacun "
            f"des {self.stats['Z'].shape[1]} capteurs de la grille de mesure.",
            self.styles["Body"]
        ))
        story.append(Spacer(1, 2 * mm))
        story.append(self._sensor_table())

        # ── Section Conclusion ───────────────────────────────
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph("Conclusion et Recommandations",
                                self.styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=self.C_BLUE, spaceAfter=6))

        badge = s["badge"]
        if badge == "Good":
            rec = (
                "La surface analysee presente une qualite conforme aux normes routieres "
                "en vigueur. Aucune intervention immediate n'est requise. "
                "Un suivi periodique (6 à 12 mois) est recommande pour maintenir cet etat."
            )
        elif badge == "Warning":
            rec = (
                "Des irregularites moderees ont ete detectees sur certaines sections. "
                "Une surveillance rapprochee est recommandee. Des travaux de maintenance "
                "preventive sont a envisager dans les 3 à 6 prochains mois "
                "afin d'eviter une degradation plus importante."
            )
        else:
            rec = (
                "Des defauts critiques ont ete identifies, depassant les seuils IRI "
                "acceptables. Une intervention urgente est requise. Les zones presentant "
                "des obstacles (points > 2 sigma) doivent etre traitees en priorite "
                "pour garantir la securite des usagers et prevenir des degradations "
                "structurelles irreversibles."
            )

        story.append(Paragraph(rec, self.styles["Body"]))

        # Tableau récapitulatif recommandations
        reco_data = [
            ["Critere", "Valeur mesuree", "Seuil acceptable", "Verdict"],
            ["Hauteur max",
             f"{s['maxv']:.2f} cm",
             "< 3.0 cm",
             "OK" if s['maxv'] < 3 else "HORS NORME"],
            ["Score qualite",
             f"{s['quality_score']} / 100",
             ">= 60 / 100",
             "OK" if s['quality_score'] >= 60 else "INSUFFISANT"],
            ["Obstacles detectes",
             str(s['nb_obstacles']),
             "< 10",
             "OK" if s['nb_obstacles'] < 10 else "ATTENTION"],
        ]

        reco_style = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), self.C_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, _hex_to_rl_color("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.4, self.C_BORDER),
            ("LINEBELOW", (0, 0), (-1, 0), 1.0, self.C_BLUE),
        ])

        # Colorer les verdicts
        for ri, row in enumerate(reco_data[1:], start=1):
            verdict = row[3]
            if "OK" in verdict:
                reco_style.add("TEXTCOLOR",  (3, ri), (3, ri), self.C_GREEN)
                reco_style.add("FONTNAME",   (3, ri), (3, ri), "Helvetica-Bold")
            else:
                reco_style.add("TEXTCOLOR",  (3, ri), (3, ri), self.C_RED)
                reco_style.add("FONTNAME",   (3, ri), (3, ri), "Helvetica-Bold")
                reco_style.add("BACKGROUND", (3, ri), (3, ri),
                               _hex_to_rl_color("#FEF2F2"))

        reco_tbl = Table(
            reco_data,
            colWidths=[4*cm, 4*cm, 4*cm, 5.5*cm],
        )
        reco_tbl.setStyle(reco_style)

        story.append(Spacer(1, 3 * mm))
        story.append(reco_tbl)

        # ── Construction finale ──────────────────────────────
        doc.build(story, onFirstPage=self._header_footer,
                  onLaterPages=self._header_footer)

        # Nettoyage des fichiers temporaires
        for f in self._tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass


# =========================================================
# ====================== WIDGETS HELPERS =================
# =========================================================

def make_shadow(blur=20, offset_y=4, opacity=30):
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, offset_y)
    shadow.setColor(QColor(15, 23, 42, opacity))
    return shadow


def h_separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #334155;")
    return line


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "—", unit: str = "",
                 sub: str = "", accent: str = "#2563EB", parent=None):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setGraphicsEffect(make_shadow(16, 3, 20))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(4)

        lbl_title = QLabel(title.upper())
        lbl_title.setStyleSheet(
            "font-size:10px; font-weight:700; color:#94A3B8; letter-spacing:1px;"
        )

        row = QHBoxLayout()
        self.lbl_value = QLabel(value)
        self.lbl_value.setStyleSheet(
            f"font-size:26px; font-weight:700; color:{accent};"
        )
        lbl_unit = QLabel(unit)
        lbl_unit.setStyleSheet("font-size:13px; color:#64748B; margin-top:10px;")
        row.addWidget(self.lbl_value)
        row.addWidget(lbl_unit)
        row.addStretch()

        self.lbl_sub = QLabel(sub)
        self.lbl_sub.setStyleSheet("font-size:11px; color:#94A3B8;")

        layout.addWidget(lbl_title)
        layout.addLayout(row)
        layout.addWidget(self.lbl_sub)

    def update(self, value: str, sub: str = ""):
        self.lbl_value.setText(value)
        self.lbl_sub.setText(sub)


class SidebarNavBtn(QPushButton):
    def __init__(self, icon: str, label: str, parent=None):
        super().__init__(f"  {icon}   {label}", parent)
        self.setObjectName("NavBtn")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(44)
        self.setCheckable(False)


# =========================================================
# ====================== 3D SURFACE ======================
# =========================================================

class Surface3DCanvas(FigureCanvasQTAgg):
    CMAP = LinearSegmentedColormap.from_list(
        "road", ["#1D4ED8", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444"]
    )

    def __init__(self):
        self.fig = Figure(facecolor="#FFFFFF", tight_layout=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax   = self.fig.add_subplot(111, projection="3d")
        self.elev = 30
        self.azim = -60
        self.Z    = np.zeros((8, 16))
        self._style_axes()
        self._draw_placeholder()

    def _style_axes(self):
        self.ax.set_facecolor("#F8FAFC")
        self.fig.patch.set_facecolor("#FFFFFF")
        for spine in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            spine.line.set_color("#E2E8F0")
        self.ax.tick_params(colors="#94A3B8", labelsize=8)
        self.ax.xaxis.label.set_color("#64748B")
        self.ax.yaxis.label.set_color("#64748B")
        self.ax.zaxis.label.set_color("#64748B")

    def _draw_placeholder(self):
        self.ax.clear()
        X, Y = np.meshgrid(range(16), range(8))
        Z = np.zeros((8, 16))
        self.ax.plot_surface(X, Y, Z, color="#E2E8F0", edgecolor="none", alpha=0.5)
        self.ax.set_title("Importer un fichier CSV pour visualiser", fontsize=11,
                          color="#94A3B8", pad=10)
        self._style_axes()
        self.ax.set_xticks([]); self.ax.set_yticks([]); self.ax.set_zticks([])
        self.draw_idle()

    def update_surface(self, Z: np.ndarray):
        self.Z = Z
        self.ax.clear()
        rows, cols = Z.shape
        X, Y = np.meshgrid(range(cols), range(rows))
        surf = self.ax.plot_surface(
            X, Y, Z, cmap=self.CMAP, edgecolor="none",
            antialiased=True, shade=True, alpha=0.95
        )
        self.fig.colorbar(surf, ax=self.ax, shrink=0.5, aspect=10,
                          label="Hauteur (mm)", pad=0.1)
        self._style_axes()
        self.ax.set_xlabel("Largeur (capteurs)", labelpad=8, fontsize=9)
        self.ax.set_ylabel("Longueur (points)", labelpad=8, fontsize=9)
        self.ax.set_zlabel("Hauteur (mm)", labelpad=8, fontsize=9)
        self.ax.set_title("Modèle surfacique 3D", fontsize=12,
                          fontweight="bold", color="#1E293B", pad=12)
        self.ax.view_init(self.elev, self.azim)
        self.ax.grid(True, color="#E2E8F0", alpha=0.6)
        self.draw_idle()

    def rotate_step(self):
        self.azim = (self.azim + 1) % 360
        self.ax.view_init(self.elev, self.azim)
        self.draw_idle()

    def view_top(self):
        self.elev, self.azim = 90, -90
        self.ax.view_init(self.elev, self.azim)
        self.draw_idle()

    def view_profile(self):
        self.elev, self.azim = 0, -90
        self.ax.view_init(self.elev, self.azim)
        self.draw_idle()

    def view_default(self):
        self.elev, self.azim = 30, -60
        self.ax.view_init(self.elev, self.azim)
        self.draw_idle()


# =========================================================
# ====================== CHARTS PANEL ====================
# =========================================================

class ChartsPanel(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(facecolor="#FFFFFF", tight_layout=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._draw_empty()

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#F8FAFC")
        ax.text(0.5, 0.5, "Aucune donnée chargée",
                ha="center", va="center", color="#94A3B8", fontsize=12,
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#E2E8F0")
        self.draw_idle()

    def update_charts(self, stats: dict):
        self.fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.45, wspace=0.35)

        Z             = stats["Z"]
        profil_long   = stats["profil_long"]
        profil_transv = stats["profil_transv"]

        BLUE  = "#2563EB"
        CYAN  = "#0EA5E9"
        GREEN = "#10B981"
        RED   = "#EF4444"
        GRAY  = "#E2E8F0"

        def _style(ax, title):
            ax.set_facecolor("#F8FAFC")
            ax.set_title(title, fontsize=8, fontweight="600", color="#1E293B", pad=2)
            ax.tick_params(colors="#64748B", labelsize=8)
            for sp in ax.spines.values(): sp.set_color(GRAY)
            ax.grid(axis="y", color=GRAY, linewidth=0.8)

        ax1 = self.fig.add_subplot(gs[0, 0])
        x1  = np.arange(len(profil_long))
        ax1.fill_between(x1, profil_long, alpha=0.15, color=BLUE)
        ax1.plot(x1, profil_long, color=BLUE, linewidth=2, marker="o", markersize=3)
        ax1.axhline(np.mean(profil_long), color=CYAN, linestyle="--",
                    linewidth=1, label="Moyenne")
        ax1.legend(fontsize=8)
        ax1.set_xlabel("Points de mesure", fontsize=8, color="#64748B")
        ax1.set_ylabel("Hauteur (mm)", fontsize=8, color="#64748B")
        _style(ax1, "Profil longitudinal")

        ax2 = self.fig.add_subplot(gs[0, 1])
        x2  = np.arange(len(profil_transv))
        ax2.fill_between(x2, profil_transv, alpha=0.15, color=GREEN)
        ax2.plot(x2, profil_transv, color=GREEN, linewidth=2, marker="s", markersize=3)
        ax2.axhline(np.mean(profil_transv), color=CYAN, linestyle="--",
                    linewidth=1, label="Moyenne")
        ax2.legend(fontsize=8)
        ax2.set_xlabel("Capteurs (largeur)", fontsize=8, color="#64748B")
        ax2.set_ylabel("Hauteur (mm)", fontsize=8, color="#64748B")
        _style(ax2, "Profil transversal")

        ax3 = self.fig.add_subplot(gs[1, 0])
        im  = ax3.imshow(Z, cmap="RdYlGn_r", aspect="auto", interpolation="nearest")
        self.fig.colorbar(im, ax=ax3, shrink=0.8, label="mm", pad=0.02)
        ax3.set_xlabel("Capteurs (largeur)", fontsize=8, color="#64748B")
        ax3.set_ylabel("Points de mesure", fontsize=8, color="#64748B")
        _style(ax3, "Carte thermique — Défauts")
        ax3.grid(False)

        ax4    = self.fig.add_subplot(gs[1, 1])
        Z_flat = Z.flatten()
        n, bins, patches = ax4.hist(Z_flat, bins=20, color=BLUE,
                                     edgecolor="white", linewidth=0.5, alpha=0.85)
        threshold = np.mean(Z_flat) + 2 * np.std(Z_flat)
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(RED if left > threshold else BLUE)
        ax4.axvline(np.mean(Z_flat), color=CYAN, linestyle="--",
                    linewidth=1.5, label="Moyenne")
        ax4.axvline(threshold, color=RED, linestyle=":",
                    linewidth=1.5, label="Seuil obstacles")
        ax4.legend(fontsize=8)
        ax4.set_xlabel("Hauteur (mm)", fontsize=8, color="#64748B")
        ax4.set_ylabel("Fréquence", fontsize=8, color="#64748B")
        _style(ax4, "Distribution des hauteurs")

        self.draw_idle()


# =========================================================
# ====================== SIDEBAR =========================
# =========================================================

class Sidebar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setStyleSheet("background:#1E293B; border-bottom:1px solid #334155;")
        header.setFixedHeight(72)
        h_lay  = QVBoxLayout(header)
        h_lay.setContentsMargins(20, 0, 20, 0)

        lbl_app = QLabel(APP_NAME)
        lbl_app.setStyleSheet(
            "font-size:17px; font-weight:700; color:#F1F5F9; letter-spacing:0.5px;"
        )
        lbl_ver = QLabel(f"v{APP_VERSION}  ·  {APP_AUTHOR}")
        lbl_ver.setStyleSheet("font-size:10px; color:#64748B;")

        h_lay.addSpacing(12)
        h_lay.addWidget(lbl_app)
        h_lay.addWidget(lbl_ver)
        layout.addWidget(header)

        nav_frame = QFrame()
        nav_frame.setStyleSheet("background:#1E293B;")
        nav_lay = QVBoxLayout(nav_frame)
        nav_lay.setContentsMargins(12, 20, 12, 10)
        nav_lay.setSpacing(4)

        section_lbl = QLabel("NAVIGATION")
        section_lbl.setStyleSheet(
            "font-size:9px; font-weight:700; color:#475569; letter-spacing:1.5px; padding-left:8px;"
        )
        nav_lay.addWidget(section_lbl)
        nav_lay.addSpacing(8)

        self.btn_import  = SidebarNavBtn("📂", "Importer CSV")
        self.btn_view3d  = SidebarNavBtn("🗻", "Vue 3D")
        self.btn_charts  = SidebarNavBtn("📊", "Graphiques")
        self.btn_report  = SidebarNavBtn("📄", "Exporter Rapport PDF")

        for btn in [self.btn_import, self.btn_view3d, self.btn_charts, self.btn_report]:
            nav_lay.addWidget(btn)

        nav_lay.addSpacing(20)

        section_lbl2 = QLabel("CAMÉRA 3D")
        section_lbl2.setStyleSheet(
            "font-size:9px; font-weight:700; color:#475569; letter-spacing:1.5px; padding-left:8px;"
        )
        nav_lay.addWidget(section_lbl2)
        nav_lay.addSpacing(8)

        self.btn_default = SidebarNavBtn("🧭", "Vue Isométrique")
        self.btn_top     = SidebarNavBtn("⬆", "Vue Haut")
        self.btn_profile = SidebarNavBtn("➡", "Vue Profil")
        self.btn_rotate  = SidebarNavBtn("🔄", "Rotation Auto")

        for btn in [self.btn_default, self.btn_top, self.btn_profile, self.btn_rotate]:
            nav_lay.addWidget(btn)

        nav_lay.addStretch()
        layout.addWidget(nav_frame)

        footer = QFrame()
        footer.setStyleSheet("background:#1E293B; border-top:1px solid #334155;")
        f_lay  = QVBoxLayout(footer)
        f_lay.setContentsMargins(20, 12, 20, 12)

        self.lbl_status = QLabel("Aucun fichier chargé")
        self.lbl_status.setStyleSheet("font-size:10px; color:#64748B;")
        self.lbl_status.setWordWrap(True)

        f_lay.addWidget(self.lbl_status)
        layout.addWidget(footer)


# =========================================================
# ====================== MAIN WINDOW =====================
# =========================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  · FR")
        self.resize(1400, 860)
        self.setMinimumSize(1100, 700)

        self.rotating       = False
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self._rotate_step)

        # Données courantes (pour export PDF)
        self._current_model = None
        self._current_stats = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar()
        root.addWidget(self.sidebar)

        self.content = QWidget()
        self.content.setStyleSheet("background:#F8FAFC;")
        content_lay = QVBoxLayout(self.content)
        content_lay.setContentsMargins(0, 0, 0, 0)
        content_lay.setSpacing(0)

        content_lay.addWidget(self._build_header())
        content_lay.addWidget(self._build_kpi_row())

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget { background: transparent; }")

        tab_3d = QWidget()
        tab_3d.setStyleSheet("background:#FFFFFF;")
        lay_3d = QVBoxLayout(tab_3d)
        lay_3d.setContentsMargins(0, 0, 0, 0)
        self.surface3d = Surface3DCanvas()
        lay_3d.addWidget(self.surface3d)
        self.tabs.addTab(tab_3d, "  🗻  Modèle 3D  ")

        tab_charts = QWidget()
        tab_charts.setStyleSheet("background:#FFFFFF;")
        lay_charts = QVBoxLayout(tab_charts)
        lay_charts.setContentsMargins(0, 0, 0, 0)
        self.charts = ChartsPanel()
        lay_charts.addWidget(self.charts)
        self.tabs.addTab(tab_charts, "  📊  Analyses  ")

        tab_data = QWidget()
        tab_data.setStyleSheet("background:#FFFFFF; padding:16px;")
        lay_data = QVBoxLayout(tab_data)
        lay_data.setContentsMargins(16, 16, 16, 16)
        self.table = self._build_table()
        lay_data.addWidget(self.table)
        self.tabs.addTab(tab_data, "  🗃  Données  ")

        content_lay.addWidget(self.tabs, stretch=1)

        self.status_bar = QStatusBar()
        self.status_bar.showMessage(
            "Prêt  ·  Importez un fichier CSV pour commencer l'analyse."
        )
        self.setStatusBar(self.status_bar)
        root.addWidget(self.content, stretch=1)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setStyleSheet(
            "background:#FFFFFF; border-bottom:1px solid #E2E8F0;"
        )
        header.setFixedHeight(64)
        lay = QHBoxLayout(header)
        lay.setContentsMargins(24, 0, 24, 0)

        self.lbl_title = QLabel("Tableau de bord — Analyse de surface routière")
        self.lbl_title.setStyleSheet(
            "font-size:16px; font-weight:700; color:#0F172A;"
        )

        self.lbl_filename = QLabel("Aucun fichier")
        self.lbl_filename.setStyleSheet("font-size:12px; color:#94A3B8;")

        self.badge_state = QLabel("—")
        self.badge_state.setObjectName("BadgeGood")
        self.badge_state.setAlignment(Qt.AlignCenter)
        self.badge_state.setFixedHeight(26)

        lay.addWidget(self.lbl_title)
        lay.addStretch()
        lay.addWidget(self.lbl_filename)
        lay.addSpacing(16)
        lay.addWidget(self.badge_state)
        return header

    def _build_kpi_row(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-bottom: 1px solid #E2E8F0;
                border-radius: 10px;
            }
            QLabel { color: #1F2937; font-size: 16px; font-weight: 600; }
            QLabel#valueLabel { color: #0F172A; font-size: 22px; font-weight: 800; }
        """)
        frame.setFixedHeight(100)
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(10, 2, 10, 2)
        lay.setSpacing(4)

        self.card_avg    = MetricCard("Hauteur Moyenne",  "—", "cm", "Valeur centrale",  "#2563EB")
        self.card_max    = MetricCard("Hauteur Maximale", "—", "cm", "Pic enregistré",   "#EF4444")
        self.card_std    = MetricCard("Écart-type",       "—", "cm", "Dispersion",       "#F59E0B")
        self.card_length = MetricCard("Longueur Estimée", "—", "m",  "Section analysée", "#0EA5E9")

        for card in [self.card_avg, self.card_max, self.card_std, self.card_length]:
            lay.addWidget(card, stretch=1)
        return frame

    def _build_table(self) -> QTableWidget:
        table = QTableWidget(0, 7)
        table.setHorizontalHeaderLabels([
            "Capteur", "Moy. (cm)", "Max (cm)", "Min (cm)",
            "Std (cm)", "Obstacles", "État"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(
            "QTableWidget { alternate-background-color: #F8FAFC; }"
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        return table

    def _connect_signals(self):
        self.sidebar.btn_import.clicked.connect(self.import_csv)
        self.sidebar.btn_view3d.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        self.sidebar.btn_charts.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        self.sidebar.btn_report.clicked.connect(self.export_report)
        self.sidebar.btn_default.clicked.connect(self.surface3d.view_default)
        self.sidebar.btn_top.clicked.connect(self.surface3d.view_top)
        self.sidebar.btn_profile.clicked.connect(self.surface3d.view_profile)
        self.sidebar.btn_rotate.clicked.connect(self.toggle_rotation)

    # ── Actions ───────────────────────────────────────────

    def import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un fichier CSV", "",
            "Fichiers CSV (*.csv);;Tous les fichiers (*)"
        )
        if not fname:
            return

        model = RoadDataModel(fname)
        Z     = model.Z

        if Z is None or Z.size == 0:
            QMessageBox.critical(self, "Erreur", "Fichier CSV invalide ou vide.")
            return

        stats = RoadAnalytics.compute(Z)
        if not stats:
            QMessageBox.warning(self, "Avertissement",
                                "Données insuffisantes pour l'analyse.")
            return

        # Stocker pour l'export PDF
        self._current_model = model
        self._current_stats = stats

        self._update_kpi(stats)
        self._update_header(model, stats)
        self._update_table(Z, stats)
        self.surface3d.update_surface(Z)
        self.charts.update_charts(stats)

        self.sidebar.lbl_status.setText(
            f"📁  {model.filename}\n🕐  {model.load_time}"
        )
        self.status_bar.showMessage(
            f"Analyse complète  ·  {model.filename}  ·  "
            f"{Z.shape[0]} × {Z.shape[1]} points  ·  {model.load_time}"
        )
        self.tabs.setCurrentIndex(0)

    def _update_kpi(self, s: dict):
        self.card_avg.update(f"{s['avg']:.2f}")
        self.card_max.update(f"{s['maxv']:.2f}", s["iri_score"])
        self.card_std.update(f"{s['std']:.2f}")
        self.card_length.update(f"{s['longueur_m']:.2f}")

    def _update_header(self, model: RoadDataModel, s: dict):
        self.lbl_filename.setText(model.filename)
        badge_map = {
            "Good":     ("BadgeGood",     s["state"]),
            "Warning":  ("BadgeWarning",  s["state"]),
            "Critical": ("BadgeCritical", s["state"]),
        }
        obj_name, text = badge_map.get(s["badge"], ("BadgeGood", "—"))
        self.badge_state.setObjectName(obj_name)
        self.badge_state.setText(f"  État : {text}  ")
        self.badge_state.style().unpolish(self.badge_state)
        self.badge_state.style().polish(self.badge_state)

    def _update_table(self, Z: np.ndarray, stats: dict):
        self.table.setRowCount(0)
        threshold = np.mean(Z) + 2 * np.std(Z)

        for i, col in enumerate(Z.T):
            col_clean = col[~np.isnan(col)]
            if col_clean.size == 0:
                continue
            avg  = float(np.mean(col_clean)) / 10
            maxv = float(np.max(col_clean))  / 10
            minv = float(np.min(col_clean))  / 10
            std  = float(np.std(col_clean))  / 10
            obs  = int(np.count_nonzero(col > threshold))
            etat = "✅ Bon" if maxv < 1 else ("⚠️ Moyen" if maxv < 3 else "❌ Critique")

            row = self.table.rowCount()
            self.table.insertRow(row)
            for j, val in enumerate([
                f"C{i+1:02d}", f"{avg:.2f}", f"{maxv:.2f}",
                f"{minv:.2f}", f"{std:.2f}", str(obs), etat
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, j, item)

    def toggle_rotation(self):
        if self.rotating:
            self.rotation_timer.stop()
            self.sidebar.btn_rotate.setText("  🔄   Rotation Auto")
        else:
            self.rotation_timer.start(16)
            self.sidebar.btn_rotate.setText("  ⏹   Arrêter Rotation")
        self.rotating = not self.rotating

    def _rotate_step(self):
        self.surface3d.rotate_step()

    def export_report(self):
        """Génère et sauvegarde le rapport PDF via ReportLab."""
        if self._current_model is None or self._current_stats is None:
            QMessageBox.warning(
                self, "Aucune donnée",
                "Veuillez d'abord importer un fichier CSV\n"
                "avant de générer un rapport PDF."
            )
            return

        # Suggérer un nom de fichier par défaut
        default_name = (
            self._current_model.filename.replace(".csv", "") +
            f"_rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le rapport PDF",
            default_name,
            "Fichiers PDF (*.pdf)"
        )
        if not save_path:
            return

        if not save_path.lower().endswith(".pdf"):
            save_path += ".pdf"

        # Génération
        self.status_bar.showMessage("Génération du rapport PDF en cours…")
        QApplication.processEvents()

        try:
            generator = PDFReportGenerator(
                save_path, self._current_model, self._current_stats
            )
            generator.generate()

            self.status_bar.showMessage(
                f"Rapport PDF exporté avec succès : {save_path}"
            )
            QMessageBox.information(
                self, "Rapport généré",
                f"Le rapport PDF a été exporté avec succès :\n\n{save_path}"
            )
        except Exception as e:
            self.status_bar.showMessage("Erreur lors de la génération du rapport.")
            QMessageBox.critical(
                self, "Erreur d'export",
                f"Une erreur est survenue lors de la génération du PDF :\n\n{e}"
            )


# =========================================================
# ====================== SPLASH ==========================
# =========================================================

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(680, 420)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        container = QFrame(self)
        container.setGeometry(0, 0, 680, 420)
        container.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 20px;
                border: 1px solid #E2E8F0;
            }
        """)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(60)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(15, 23, 42, 50))
        container.setGraphicsEffect(shadow)

        lay = QVBoxLayout(container)
        lay.setAlignment(Qt.AlignCenter)
        lay.setContentsMargins(60, 40, 60, 40)
        lay.setSpacing(0)

        badge = QLabel("🛣")
        badge.setStyleSheet("font-size:48px;")
        badge.setAlignment(Qt.AlignCenter)

        title = QLabel(APP_NAME)
        title.setStyleSheet(
            "font-size:30px; font-weight:700; color:#0F172A; letter-spacing:-0.5px;"
        )
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Analyse de Qualité de Surface Routière")
        subtitle.setStyleSheet("font-size:14px; color:#64748B; margin-top:4px;")
        subtitle.setAlignment(Qt.AlignCenter)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color:#E2E8F0; margin:24px 0;")

        self.lbl_step = QLabel("Initialisation du système…")
        self.lbl_step.setStyleSheet("font-size:12px; color:#94A3B8;")
        self.lbl_step.setAlignment(Qt.AlignCenter)

        prog_bg = QFrame()
        prog_bg.setFixedSize(480, 6)
        prog_bg.setStyleSheet("background:#E2E8F0; border-radius:3px;")

        self.prog_bar = QFrame(prog_bg)
        self.prog_bar.setFixedSize(0, 6)
        self.prog_bar.setStyleSheet("""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #2563EB, stop:1 #0EA5E9);
            border-radius:3px;
        """)

        self.lbl_pct = QLabel("0 %")
        self.lbl_pct.setStyleSheet(
            "font-size:13px; font-weight:600; color:#2563EB; margin-top:10px;"
        )
        self.lbl_pct.setAlignment(Qt.AlignCenter)

        version_lbl = QLabel(f"v{APP_VERSION}  ·  Mémoire d'ingénieur")
        version_lbl.setStyleSheet("font-size:10px; color:#CBD5E1; margin-top:20px;")
        version_lbl.setAlignment(Qt.AlignCenter)

        lay.addWidget(badge)
        lay.addSpacing(8)
        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addWidget(divider)
        lay.addWidget(self.lbl_step)
        lay.addSpacing(16)
        lay.addWidget(prog_bg, alignment=Qt.AlignCenter)
        lay.addWidget(self.lbl_pct)
        lay.addStretch()
        lay.addWidget(version_lbl)

        self.setWindowOpacity(0)
        self.fade = QPropertyAnimation(self, b"windowOpacity")
        self.fade.setDuration(400)
        self.fade.setStartValue(0)
        self.fade.setEndValue(1)
        self.fade.start()

        self._steps = [
            "Chargement des modules de traitement…",
            "Initialisation du moteur graphique 3D…",
            "Préparation des algorithmes d'analyse…",
            "Démarrage de l'interface…",
        ]
        self._step_i = 0
        self._step_timer = QTimer()
        self._step_timer.timeout.connect(self._next_step)
        self._step_timer.start(700)

        self.anim = QPropertyAnimation(self.prog_bar, b"minimumWidth")
        self.anim.setDuration(3200)
        self.anim.setStartValue(0)
        self.anim.setEndValue(480)
        self.anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.anim.valueChanged.connect(self._update_pct)
        self.anim.finished.connect(self._finish)
        self.anim.start()

    def _next_step(self):
        if self._step_i < len(self._steps):
            self.lbl_step.setText(self._steps[self._step_i])
            self._step_i += 1

    def _update_pct(self, val):
        pct = int((val / 480) * 100)
        self.lbl_pct.setText(f"{pct} %")

    def _finish(self):
        self._step_timer.stop()
        self.lbl_step.setText("Démarrage…")
        fade_out = QPropertyAnimation(self, b"windowOpacity")
        fade_out.setDuration(400)
        fade_out.setStartValue(1)
        fade_out.setEndValue(0)
        fade_out.finished.connect(self._launch)
        fade_out.start()
        self._fade_out = fade_out

    def _launch(self):
        self.close()
        self.main = MainWindow()
        self.main.show()


# =========================================================
# ====================== ENTRY POINT =====================
# =========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    splash = SplashScreen()
    splash.show()
    sys.exit(app.exec())