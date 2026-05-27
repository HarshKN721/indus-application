import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTabWidget, 
                             QSlider, QMessageBox, QLineEdit, QListWidget, QInputDialog, QDialog, QFormLayout)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

from core.vision import VisionProcessor
from core.analysis import DataAnalyzer
from data_fetcher.kaggle_client import KaggleScraper

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Beam Diagnostics Suite")
        self.resize(1200, 900)
        
        self.is_dark_mode = False
        self.apply_theme()

        # Initialize core components
        self.vision_processor = None # Delayed initialization to allow proxy setup
        self.analyzer = DataAnalyzer()
        self.scraper = KaggleScraper()
        
        self.current_image_data = None
        self.active_rois = []

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        
        # Top Toolbar (Proxy & Theme)
        self.setup_toolbar()

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.setup_vision_tab()
        self.setup_mesh_tab()
        self.setup_math_tab()
        self.setup_scraping_tab()

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
        # Dialog to capture proxy details
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
            ip = ip_input.text().strip()
            port = port_input.text().strip()
            user = user_input.text().strip()
            pw = pass_input.text().strip()
            
            if ip and port:
                if user and pw:
                    proxy_url = f"http://{user}:{pw}@{ip}:{port}"
                else:
                    proxy_url = f"http://{ip}:{port}"
                
                os.environ['HTTP_PROXY'] = proxy_url
                os.environ['HTTPS_PROXY'] = proxy_url
                QMessageBox.information(self, "Proxy Set", "Proxy credentials applied to environment.")

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()

    def apply_theme(self):
        # Windows 11 Style Gradients and Modern UI elements
        if self.is_dark_mode:
            bg_color = "#202020"
            text_color = "#ffffff"
            pane_color = "#2d2d2d"
            tab_inactive = "#333333"
            btn_gradient = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #005a9e, stop:1 #004070)"
            btn_hover = "#0078d4"
        else:
            bg_color = "#fafafa"
            text_color = "#1f1f1f"
            pane_color = "#ffffff"
            tab_inactive = "#f0f0f0"
            btn_gradient = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0078d4, stop:1 #0060a0)"
            btn_hover = "#005a9e"

        self.setStyleSheet(f"""
            QWidget {{ background-color: {bg_color}; color: {text_color}; font-family: 'Segoe UI', sans-serif; font-size: 14px; }}
            QPushButton {{ 
                background: {btn_gradient}; 
                color: white; 
                border-radius: 6px; 
                padding: 8px 16px; 
                font-weight: 600; 
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            QPushButton:hover {{ background: {btn_hover}; }}
            QTabWidget::pane {{ border: 1px solid #a0a0a0; background: {pane_color}; border-radius: 8px; }}
            QTabBar::tab {{ background: {tab_inactive}; padding: 10px 20px; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
            QTabBar::tab:selected {{ background: {pane_color}; border-bottom: 2px solid #0078d4; font-weight: bold; }}
            QLineEdit {{ padding: 6px; border: 1px solid #ccc; border-radius: 4px; background: {pane_color}; color: {text_color}; }}
            QListWidget {{ background: {pane_color}; color: {text_color}; border: 1px solid #ccc; border-radius: 4px; }}
        """)

    def setup_vision_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        control_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.load_image)
        
        self.analyze_btn = QPushButton("Analyze Selected ROI")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setStyleSheet("background-color: #107c10;") 
        
        # New Scale Slider Features
        self.scale_label = QLabel("Mesh Resolution: 50%")
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 100)
        self.scale_slider.setValue(50) # Default to 50% for speed
        self.scale_slider.setFixedWidth(150)
        self.scale_slider.valueChanged.connect(self.update_scale_label)

        self.info_btn = QPushButton("?")
        self.info_btn.setFixedSize(36, 36)
        self.info_btn.clicked.connect(self.show_scale_info)
        
        self.status_label = QLabel("Status: Waiting for input...")
        
        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.scale_label)
        control_layout.addWidget(self.scale_slider)
        control_layout.addWidget(self.info_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        layout.addWidget(self.status_label)

        pg.setConfigOptions(imageAxisOrder='row-major')
        self.image_widget = pg.PlotWidget()
        self.image_widget.invertY(True)
        
        # Hiding the axes so it looks like an image, not a graph
        self.image_widget.hideAxis('bottom')
        self.image_widget.hideAxis('left')
        
        layout.addWidget(self.image_widget)
        
        self.plot_item = self.image_widget.getPlotItem()
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)
        
        self.tabs.addTab(tab, "1. Image & ROI")

    def update_scale_label(self, value):
        self.scale_label.setText(f"Mesh Resolution: {value}%")

    def show_scale_info(self):
        # Pop-up explaining the slider logic
        QMessageBox.information(
            self, 
            "Resolution Scale Information", 
            "LEFT SIDE (Lower %): Downsizes the image. This processes much faster and smooths out noise, but reduces detail.\n\n"
            "RIGHT SIDE (100%): Original scale. Provides maximum detail but processing the 3D mesh will be significantly slower."
        )

    def setup_mesh_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.mesh_widget = gl.GLViewWidget()
        self.mesh_widget.opts['distance'] = 200
        
        dummy_z = np.zeros((2, 2))
        # Change shader to 'heightColor' for a much cleaner, Plotly-style aesthetic
        self.mesh_surface = gl.GLSurfacePlotItem(z=dummy_z, shader='heightColor')
        self.mesh_widget.addItem(self.mesh_surface)
        
        layout.addWidget(self.mesh_widget)
        self.tabs.addTab(tab, "2. Mesh Analysis")

    def setup_math_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.math_widget = pg.PlotWidget(title="Gaussian Fit Profile")
        self.math_widget.setBackground('w')
        self.math_widget.showGrid(x=True, y=True)
        layout.addWidget(self.math_widget)
        
        self.roc_widget = pg.PlotWidget(title="Secondary Math / ROC")
        self.roc_widget.setBackground('w')
        layout.addWidget(self.roc_widget)
        self.tabs.addTab(tab, "3. Mathematical Plots")

    def setup_scraping_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Credentials Layout
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
        
        # Search Layout
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
        user = self.k_user.text().strip()
        key = self.k_key.text().strip()
        if user and key:
            success, msg = self.scraper.authenticate_api(user, key)
            QMessageBox.information(self, "Kaggle Auth", msg)
        else:
            QMessageBox.warning(self, "Input Error", "Provide both Username and API Key.")

    def run_kaggle_search(self):
        query = self.search_input.text().strip()
        if not query:
            return
            
        self.results_list.clear()
        self.results_list.addItem("Searching...")
        self.repaint()
        
        results = self.scraper.search_datasets(query)
        self.results_list.clear()
        
        for res in results:
            self.results_list.addItem(res)

    def load_image(self):
        # Initialize YOLOv8 late so proxy settings have time to apply
        if self.vision_processor is None:
            self.vision_processor = VisionProcessor()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.status_label.setText("Processing YOLOv8...")
            self.repaint()
            
            self.current_image_data, boxes = self.vision_processor.detect_objects(file_path)
            self.image_item.setImage(self.current_image_data)
            
            for item in self.active_rois:
                self.plot_item.removeItem(item)
            self.active_rois.clear()
            
            for (x, y, w, h, conf) in boxes:
                roi = pg.ROI(pos=[x, y], size=[w, h], pen=pg.mkPen('r', width=3))
                roi.addScaleHandle([1, 1], [0, 0])
                roi.addScaleHandle([0, 0], [1, 1])
                self.plot_item.addItem(roi)
                
                text = pg.TextItem(f"Conf: {conf:.2f}", color=(255, 0, 0), anchor=(0, 1))
                text.setParentItem(roi)
                self.active_rois.append(roi)
                
            self.status_label.setText("Adjust the ROI box and click 'Analyze Selected ROI'")

    def run_analysis(self):
        if not self.active_rois or self.current_image_data is None:
            self.status_label.setText("Error: No image or ROI available.")
            return

        self.status_label.setText("Analyzing...")
        self.repaint()

        target_roi = self.active_rois[0]
        region = target_roi.getArrayRegion(self.current_image_data, self.image_item)
        scale_percent = self.scale_slider.value()

        # 1. Mesh Processing
        z_data = self.analyzer.prepare_mesh_data(region, scale_percent)
        x_shift, y_shift = z_data.shape[0] / 2, z_data.shape[1] / 2
        
        # Dynamically calculate RGBA colors based on Z-height
        z_min, z_max = z_data.min(), z_data.max()
        z_norm = (z_data - z_min) / (z_max - z_min + 1e-16) # Normalize between 0 and 1
        
        colors = np.empty(z_data.shape + (4,), dtype=np.float32)
        colors[..., 0] = z_norm  # Red channel peaks at high values
        colors[..., 1] = 1.0 - np.abs(2.0 * z_norm - 1.0)  # Green peaks in the middle
        colors[..., 2] = 1.0 - z_norm  # Blue channel peaks at low values
        colors[..., 3] = 1.0  # Alpha (Opacity)

        self.mesh_surface.setData(z=z_data, colors=colors)
        self.mesh_surface.opts['computeNormals'] = False
        
        self.mesh_surface.resetTransform()
        self.mesh_surface.translate(-x_shift, -y_shift, 0)
        
        # 2. Gaussian Processing
        x_data, raw_profile, fit_profile = self.analyzer.fit_gaussian_profile(region)
        self.math_widget.clear()
        self.math_widget.plot(x_data, raw_profile, pen=pg.mkPen('b', width=2), name="Raw Data")
        self.math_widget.plot(x_data, fit_profile, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2), name="Gaussian Fit")

        self.status_label.setText("Analysis Complete. View the other tabs.")
        self.tabs.setCurrentIndex(1)