from typing import List, Dict
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field
import requests

from typing import Any, Dict, List, Mapping, Optional

import requests
from pydantic import Extra, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
import json

class HuggingFaceEndpointEmbeddings(BaseModel, Embeddings):
    """Hugging Face Embeddings Endpoint
    To use, you should have the ``huggingface_hub`` python package installed, and the
    environment variable ``HUGGINGFACEHUB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.
    
    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEndpointEmbeddings
            endpoint_url = (
                "https://abcdefghijklmnop.us-east-1.aws.endpoints.huggingface.cloud"
            )
            hf = HuggingFaceEndpointEmbeddings(
                endpoint_url=endpoint_url,
                huggingfacehub_api_token="my-api-key"
            )
    """
    endpoint_url: str = ""
    """Endpoint URL to use."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""
    huggingfacehub_api_token: Optional[str] = None

    def __init__(self, endpoint_name = None, api_token= None) -> None:
        self.endpoint_name = endpoint_name
        self.api_token = api_token

    def embed_query(self, text: str) -> List[float]:
        """Query Hugging Face Inference Endpoint for embeddings

        Args:
            inputs (List[str]): List of texts to embed
        Raises:
            e: Error connecting to Hugging Face API
            Exception: error querying Hugging Face API

        Returns:
            List[List[float]]: List of embeddings
        """
        headers: Dict[str, str] = {
        "Authorization": f"Bearer {self.api_token}",
        "Content-Type": "application/json"
        }
        query = json.dumps({"inputs": text})
        try:
            response = requests.request("POST", self.endpoint_name, headers=headers, data=query)
        except Exception as e:
            raise e

        if response.status_code != 200:
            raise Exception(response.text)
        return json.loads(response.content.decode("utf-8"))['embeddings']
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Query Hugging Face Inference Endpoint for multiple embeddings

        Args:
            inputs (List[str]): List of texts to embed
        Raises:
            e: Error connecting to Hugging Face API
            Exception: error querying Hugging Face API

        Returns:
            List[List[float]]: List of embeddings
        """
        headers: Dict[str, str] = {
        "Authorization": f"Bearer {self.api_token}",
        "Content-Type": "application/json"
        }
        query = json.dumps({"inputs": texts})
        try:
            response = requests.request("POST", self.endpoint_name, headers=headers, data=query)
        except Exception as e:
            raise e

        if response.status_code != 200:
            raise Exception(response.text)
        return json.loads(response.content.decode("utf-8"))['embeddings']


