import numpy as np
from scipy.optimize import curve_fit
import cv2

class DataAnalyzer:
    @staticmethod
    def gaussian(x, a, mu, sigma, c):
        return a * np.exp(-((x - mu)**2) / (2 * sigma**2)) + c

    def prepare_mesh_data(self, image_crop, scale_percent=100):
        image_crop = np.clip(image_crop, 0, 255).astype(np.uint8)

        if len(image_crop.shape) == 3:
            gray_crop = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
        else:
            gray_crop = image_crop
            
        # Performance fix: Strictly cap the maximum number of vertices
        max_dim = max(10, int(150 * (scale_percent / 100.0)))
        
        h, w = gray_crop.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            gray_crop = cv2.resize(gray_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        smoothed_crop = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        z_data = np.array(smoothed_crop, dtype=float) * 0.15 
        
        return z_data

    def fit_gaussian_profile(self, image_crop):
        image_crop = np.clip(image_crop, 0, 255).astype(np.uint8)
        if len(image_crop.shape) == 3:
            gray_crop = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
        else:
            gray_crop = image_crop

        y_profile = np.mean(gray_crop, axis=0)
        x_data = np.arange(len(y_profile))

        a_guess = np.max(y_profile) - np.min(y_profile)
        mu_guess = x_data[np.argmax(y_profile)]
        sigma_guess = max(len(x_data) / 4.0, 1.0) 
        c_guess = np.min(y_profile)

        try:
            popt, _ = curve_fit(
                self.gaussian, 
                x_data, 
                y_profile, 
                p0=[a_guess, mu_guess, sigma_guess, c_guess],
                maxfev=2000
            )
            y_fit = self.gaussian(x_data, *popt)
            # popt contains: [Amplitude, Center (mu), Width (sigma), Baseline (c)]
            metrics = {
                "amplitude": popt[0],
                "centroid": popt[1],
                "sigma": abs(popt[2])
            }
            return x_data, y_profile, y_fit, metrics
        except RuntimeError:
            return x_data, y_profile, np.full_like(x_data, c_guess), None

    def find_auto_roi(self, image_data):
        """Automatically detects the brightest region (beam) and returns a bounding box [x, y, w, h]"""
        # Convert to grayscale
        if len(image_data.shape) == 3:
            gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_data.copy()

        # Blur and threshold to isolate the beam
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Determine a dynamic threshold based on max intensity
        max_val = np.max(blurred)
        thresh_val = max(50, max_val * 0.5) # Threshold at 50% of max brightness
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback if no beam found: center square
            h, w = gray.shape
            return [w//2 - 50, h//2 - 50, 100, 100]
            
        # Get the largest contour (assuming it's the primary beam)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add slight padding (e.g., 20 pixels) to ensure the whole Gaussian tail is captured
        pad = 20
        h_img, w_img = gray.shape
        x_padded = max(0, x - pad)
        y_padded = max(0, y - pad)
        w_padded = min(w_img - x_padded, w + 2*pad)
        h_padded = min(h_img - y_padded, h + 2*pad)
        
        return [x_padded, y_padded, w_padded, h_padded]