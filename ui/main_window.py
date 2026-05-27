import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTabWidget, 
                             QSlider, QMessageBox, QLineEdit, QListWidget, QDialog, QFormLayout)
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# Matplotlib Imports for the 3D Canvas
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.analysis import DataAnalyzer
from data_fetcher.kaggle_client import KaggleScraper

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Beam Diagnostics Suite")
        self.resize(1200, 900)
        
        self.is_dark_mode = False
        self.apply_theme()

        # Notice VisionProcessor is gone. We only need the math and scraper now.
        self.analyzer = DataAnalyzer()
        self.scraper = KaggleScraper()
        
        self.current_image_data = None
        self.active_rois = []

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        
        self.setup_toolbar()
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.setup_vision_tab()
        self.setup_mesh_tab()
        self.setup_math_tab()
        self.setup_scraping_tab()

    # ... [Keep your existing setup_toolbar, configure_proxy, toggle_theme, and apply_theme methods here] ...
    def setup_toolbar(self):
        toolbar_layout = QHBoxLayout()
        self.proxy_btn = QPushButton("Configure Network Proxy")
        self.proxy_btn.clicked.connect(self.configure_proxy)
        self.theme_btn = QPushButton("Toggle Dark Mode")
        self.theme_btn.clicked.connect(self.toggle_theme)
        toolbar_layout.addWidget(self.proxy_btn)
        toolbar_layout.addWidget(self.theme_btn)
        toolbar_layout.addStretch()
        self.main_layout.addLayout(toolbar_layout)

    def configure_proxy(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Proxy Authentication")
        layout = QFormLayout(dialog)
        ip_input = QLineEdit()
        port_input = QLineEdit()
        user_input = QLineEdit()
        pass_input = QLineEdit()
        pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addRow("Proxy IP/Host:", ip_input)
        layout.addRow("Port:", port_input)
        layout.addRow("Username (Optional):", user_input)
        layout.addRow("Password (Optional):", pass_input)
        submit_btn = QPushButton("Apply Proxy")
        submit_btn.clicked.connect(dialog.accept)
        layout.addRow(submit_btn)
        
        if dialog.exec():
            ip, port = ip_input.text().strip(), port_input.text().strip()
            user, pw = user_input.text().strip(), pass_input.text().strip()
            if ip and port:
                proxy_url = f"http://{user}:{pw}@{ip}:{port}" if user and pw else f"http://{ip}:{port}"
                os.environ['HTTP_PROXY'] = proxy_url
                os.environ['HTTPS_PROXY'] = proxy_url
                QMessageBox.information(self, "Proxy Set", "Proxy credentials applied.")

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()

    def apply_theme(self):
        if self.is_dark_mode:
            bg_color, text_color, pane_color, tab_inactive = "#202020", "#ffffff", "#2d2d2d", "#333333"
            btn_gradient, btn_hover = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #005a9e, stop:1 #004070)", "#0078d4"
        else:
            bg_color, text_color, pane_color, tab_inactive = "#fafafa", "#1f1f1f", "#ffffff", "#f0f0f0"
            btn_gradient, btn_hover = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0078d4, stop:1 #0060a0)", "#005a9e"

        self.setStyleSheet(f"""
            QWidget {{ background-color: {bg_color}; color: {text_color}; font-family: 'Segoe UI', sans-serif; font-size: 14px; }}
            QPushButton {{ background: {btn_gradient}; color: white; border-radius: 6px; padding: 8px 16px; font-weight: 600; border: 1px solid rgba(255, 255, 255, 0.1);}}
            QPushButton:hover {{ background: {btn_hover}; }}
            QTabWidget::pane {{ border: 1px solid #a0a0a0; background: {pane_color}; border-radius: 8px; }}
            QTabBar::tab {{ background: {tab_inactive}; padding: 10px 20px; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
            QTabBar::tab:selected {{ background: {pane_color}; border-bottom: 2px solid #0078d4; font-weight: bold; }}
            QLineEdit, QListWidget {{ padding: 6px; border: 1px solid #ccc; border-radius: 4px; background: {pane_color}; color: {text_color}; }}
        """)

    def setup_vision_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        control_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.load_image)
        
        self.add_roi_btn = QPushButton("Draw Manual ROI")
        self.add_roi_btn.clicked.connect(self.add_manual_roi)
        self.add_roi_btn.setEnabled(False) # Disabled until image is loaded
        
        self.analyze_btn = QPushButton("Analyze Beam")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setStyleSheet("background-color: #107c10;") 
        
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 100)
        self.scale_slider.setValue(50)
        self.scale_slider.setFixedWidth(150)
        
        self.status_label = QLabel("Status: Waiting for image...")
        
        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.add_roi_btn)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Mesh Res:"))
        control_layout.addWidget(self.scale_slider)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        layout.addWidget(self.status_label)

        pg.setConfigOptions(imageAxisOrder='row-major')
        self.image_widget = pg.PlotWidget()
        self.image_widget.invertY(True)
        self.image_widget.hideAxis('bottom')
        self.image_widget.hideAxis('left')
        
        layout.addWidget(self.image_widget)
        
        self.plot_item = self.image_widget.getPlotItem()
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)
        
        self.tabs.addTab(tab, "1. Image & ROI")

    def setup_mesh_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # --- NEW MATPLOTLIB 3D INTEGRATION ---
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Make the background match our theme
        self.fig.patch.set_facecolor('#fafafa')
        self.ax.set_facecolor('#fafafa')
        self.ax.set_title("Interactive Beam Intensity Topography")
        
        layout.addWidget(self.canvas)
        self.tabs.addTab(tab, "2. Interactive Mesh")

    def setup_math_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Diagnostic Metrics Label
        self.metrics_label = QLabel("Beam Metrics: Pending Analysis...")
        self.metrics_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0067c0; padding: 10px;")
        layout.addWidget(self.metrics_label)
        
        # We removed the ROC curve and gave Gaussian the full window
        self.math_widget = pg.PlotWidget(title="Gaussian Beam Profile")
        self.math_widget.setBackground('w')
        self.math_widget.showGrid(x=True, y=True)
        layout.addWidget(self.math_widget)
        
        self.tabs.addTab(tab, "3. Diagnostic Math")

    # ... [Keep your existing setup_scraping_tab and authenticate_kaggle methods here] ...
    def setup_scraping_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        cred_layout = QHBoxLayout()
        self.k_user = QLineEdit()
        self.k_user.setPlaceholderText("Kaggle Username")
        self.k_key = QLineEdit()
        self.k_key.setPlaceholderText("Kaggle API Key")
        self.k_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.auth_btn = QPushButton("Authenticate API")
        self.auth_btn.clicked.connect(self.authenticate_kaggle)
        cred_layout.addWidget(self.k_user)
        cred_layout.addWidget(self.k_key)
        cred_layout.addWidget(self.auth_btn)
        
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search Datasets (e.g., 'laser profile')")
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.run_kaggle_search)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_btn)
        
        self.results_list = QListWidget()
        layout.addLayout(cred_layout)
        layout.addLayout(search_layout)
        layout.addWidget(self.results_list)
        self.tabs.addTab(tab, "4. Data Scraper")

    def authenticate_kaggle(self):
        user, key = self.k_user.text().strip(), self.k_key.text().strip()
        if user and key:
            success, msg = self.scraper.authenticate_api(user, key)
            QMessageBox.information(self, "Kaggle Auth", msg)
        else:
            QMessageBox.warning(self, "Input Error", "Provide both Username and API Key.")
            
    def run_kaggle_search(self):
        query = self.search_input.text().strip()
        if query:
            self.results_list.clear()
            self.results_list.addItem("Searching...")
            self.repaint()
            results = self.scraper.search_datasets(query)
            self.results_list.clear()
            for res in results:
                self.results_list.addItem(res)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            # Replaced YOLO with simple OpenCV loading
            img = cv2.imread(file_path)
            self.current_image_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image_item.setImage(self.current_image_data)
            
            for item in self.active_rois:
                self.plot_item.removeItem(item)
            self.active_rois.clear()
            
            self.add_roi_btn.setEnabled(True)
            self.status_label.setText("Image loaded. Click 'Draw Manual ROI' to select the beam.")

    def add_manual_roi(self):
        # Spawns a box in the middle of the image for the user to drag
        h, w, _ = self.current_image_data.shape
        roi = pg.ROI(pos=[w//2 - 50, h//2 - 50], size=[100, 100], pen=pg.mkPen('r', width=3))
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        self.plot_item.addItem(roi)
        self.active_rois.append(roi)
        self.status_label.setText("Adjust the red box over the beam and click 'Analyze Beam'.")

    def run_analysis(self):
        if not self.active_rois or self.current_image_data is None:
            self.status_label.setText("Error: Draw an ROI first.")
            return

        self.status_label.setText("Processing Mathematics...")
        self.repaint()

        target_roi = self.active_rois[0]
        region = target_roi.getArrayRegion(self.current_image_data, self.image_item)
        scale_percent = self.scale_slider.value()

        # 1. Matplotlib Mesh Processing
        z_data = self.analyzer.prepare_mesh_data(region, scale_percent)
        
        self.ax.clear()
        self.fig.patch.set_facecolor('#202020' if self.is_dark_mode else '#fafafa')
        self.ax.set_facecolor('#202020' if self.is_dark_mode else '#fafafa')
        
        # Create X and Y grids for Matplotlib
        x = np.arange(z_data.shape[1])
        y = np.arange(z_data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Plot the interactive surface
        surf = self.ax.plot_surface(X, Y, z_data, cmap='jet', edgecolor='none')
        self.ax.set_xlabel('X Pixel Space')
        self.ax.set_ylabel('Y Pixel Space')
        self.ax.set_zlabel('Intensity')
        self.ax.set_title("Interactive Beam Intensity Topography", color='white' if self.is_dark_mode else 'black')
        
        # Update the canvas drawing
        self.canvas.draw()
        
        # 2. Gaussian Processing & Metrics
        x_data, raw_profile, fit_profile, metrics = self.analyzer.fit_gaussian_profile(region)
        
        self.math_widget.clear()
        self.math_widget.plot(x_data, raw_profile, pen=pg.mkPen('b', width=2), name="Raw Profile")
        self.math_widget.plot(x_data, fit_profile, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2), name="Gaussian Fit")

        # Update the diagnostic labels
        if metrics:
            self.metrics_label.setText(
                f"Beam Diagnostics | Peak Intensity: {metrics['amplitude']:.2f} | "
                f"Centroid (X): {metrics['centroid']:.2f} | Beam Width (σ): {metrics['sigma']:.2f}"
            )
        else:
            self.metrics_label.setText("Beam Diagnostics: Failed to converge fit. Data is too noisy.")

        self.status_label.setText("Analysis Complete.")
        self.tabs.setCurrentIndex(1)