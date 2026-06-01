import os
import sys

class KaggleScraper:
    def __init__(self):
        self.api = None
        self.authenticated = False

    def authenticate_api(self, username, key):
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
        try:
            modules_to_remove = [mod for mod in sys.modules if mod.startswith('kaggle')]
            for mod in modules_to_remove:
                del sys.modules[mod]
                
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            self.api = KaggleApi()
            self.api.authenticate()
            self.api.dataset_list(search='test', max_size=1)
            
            self.authenticated = True
            return True, "Authentication successful. You can now search."
        except Exception as e:
            self.authenticated = False
            return False, f"Auth Error: {str(e)}"

    def search_datasets(self, query, max_results=10):
        if not self.authenticated:
            return ["Authentication Required: Please enter your Kaggle credentials first."]
        
        try:
            datasets = self.api.dataset_list(search=query, sort_by='hottest')
            results = []
            for i, dataset in enumerate(datasets):
                if i >= max_results:
                    break
                
                # Safely fetch attributes to prevent crashes if Kaggle updates their API
                title = getattr(dataset, 'title', 'Unknown Title')
                ref = getattr(dataset, 'ref', 'Unknown Reference')
                updated = getattr(dataset, 'lastUpdated', 'Unknown Date')
                
                # Use totalBytes and safely convert it to Megabytes for the UI
                size_bytes = getattr(dataset, 'totalBytes', 0)
                if size_bytes:
                    size_mb = f"{int(size_bytes) / (1024 * 1024):.2f} MB"
                else:
                    size_mb = "Unknown Size"

                results.append(
                    f"Title: {title}\n"
                    f"Ref: {ref}\n"
                    f"Size: {size_mb}\n"
                    f"Updated: {updated}"
                )
            
            if not results:
                return ["No datasets found."]
            return results
        except Exception as e:
            return [f"Search Error: {str(e)}"]

    def download_dataset(self, dataset_ref, download_path):
        """Downloads and unzips the dataset to the specified path."""
        if not self.authenticated:
            return False, "Not authenticated."
        try:
            # unzip=True automatically extracts the files from the Kaggle zip archive
            self.api.dataset_download_files(dataset_ref, path=download_path, unzip=True)
            return True, f"Successfully downloaded and extracted to:\n{download_path}"
        except Exception as e:
            return False, f"Download failed: {str(e)}"