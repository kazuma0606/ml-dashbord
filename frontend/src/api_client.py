"""API client for communicating with the FastAPI backend."""
import time
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import settings


class MLAPIClient:
    """Client for ML Dashboard backend API with retry logic."""
    
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None):
        """Initialize API client.
        
        Args:
            base_url: Backend API base URL (defaults to settings.api_base_url)
            timeout: Request timeout in seconds (defaults to settings.api_timeout)
        """
        self.base_url = base_url or settings.api_base_url
        self.timeout = timeout or settings.api_timeout
        
        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> requests.Response:
        """Make HTTP request with error handling.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: On request failure
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            raise requests.RequestException(
                f"API request failed: {method} {endpoint} - {str(e)}"
            ) from e
    
    def get_datasets(self) -> List[str]:
        """Get list of available datasets.
        
        Returns:
            List of dataset names
        """
        response = self._make_request("GET", "/api/datasets")
        data = response.json()
        return data.get("datasets", [])
    
    def get_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset details.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset information including metadata
        """
        response = self._make_request("GET", f"/api/datasets/{dataset_name}")
        return response.json()
    
    def get_dataset_preview(
        self, 
        dataset_name: str, 
        n_rows: int = 10
    ) -> Dict[str, Any]:
        """Get dataset preview.
        
        Args:
            dataset_name: Name of the dataset
            n_rows: Number of rows to preview
            
        Returns:
            Preview data with features and target
        """
        response = self._make_request(
            "GET", 
            f"/api/datasets/{dataset_name}/preview",
            params={"n_rows": n_rows}
        )
        return response.json()
    
    def train_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a model with given configuration.
        
        Args:
            config: Training configuration including:
                - dataset_name: str
                - test_size: float
                - random_state: int
                - model_type: str
                - hyperparameters: dict
                
        Returns:
            Training results including metrics and model_id
        """
        response = self._make_request("POST", "/api/train", json=config)
        return response.json()
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get model information.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information
        """
        response = self._make_request("GET", f"/api/models/{model_id}")
        return response.json()
    
    def export_model(self, model_id: str) -> bytes:
        """Export trained model as pickle file.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Pickled model bytes
        """
        response = self._make_request("GET", f"/api/models/{model_id}/export")
        return response.content
    
    def save_experiment(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save experiment record.
        
        Args:
            experiment_data: Experiment data to save
            
        Returns:
            Saved experiment record with ID
        """
        response = self._make_request("POST", "/api/experiments", json=experiment_data)
        return response.json()
    
    def get_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiment records.
        
        Returns:
            List of experiment records
        """
        response = self._make_request("GET", "/api/experiments")
        data = response.json()
        return data.get("experiments", [])
    
    def clear_experiments(self) -> Dict[str, Any]:
        """Clear all experiment records.
        
        Returns:
            Success message
        """
        response = self._make_request("DELETE", "/api/experiments")
        return response.json()
