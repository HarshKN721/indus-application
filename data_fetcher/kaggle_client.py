import os
import sys

class KaggleScraper:
    def __init__(self):
        self.api = None
        self.authenticated = False

    def authenticate_api(self, username, key):
        # 1. Inject credentials into the environment
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
        try:
            # 2. Forcibly clear Kaggle from Python's memory cache.
            # This prevents Python from remembering a previously failed auth state
            # and forces it to read the new environment variables.
            modules_to_remove = [mod for mod in sys.modules if mod.startswith('kaggle')]
            for mod in modules_to_remove:
                del sys.modules[mod]
                
            # 3. Import and Authenticate freshly
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            self.api = KaggleApi()
            self.api.authenticate()
            
            # 4. Quick silent test to confirm the credentials are truly valid
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
            datasets = self.api.dataset_list(search=query, max_size=500)
            results = []
            for i, dataset in enumerate(datasets):
                if i >= max_results:
                    break
                results.append(f"Title: {dataset.title}\nRef: {dataset.ref}\nDownloads: {dataset.downloadCount}")
            
            if not results:
                return ["No datasets found."]
            return results
        except Exception as e:
            return [f"Search Error: {str(e)}"]