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
        # 100% scale = max 150 pixels. 10% scale = max 15 pixels.
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