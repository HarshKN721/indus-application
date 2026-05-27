import os

class KaggleScraper:
    def __init__(self):
        self.api = None
        self.authenticated = False

    def authenticate_api(self, username, key):
        # Inject credentials into the environment variables for this session only
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
        try:
            # LAZY IMPORT: Prevents Kaggle from crashing the app on startup
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.api = KaggleApi()
            self.api.authenticate()
            self.authenticated = True
            return True, "Authentication successful."
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