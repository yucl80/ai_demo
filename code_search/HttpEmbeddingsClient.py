from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, SecretStr
from openai import OpenAI

class HttpEmbeddingsClient(BaseModel, Embeddings):
    """Embed texts using the API.   
    """
 
    api_url: Optional[str] = "http://localhost:8000/v1/"
    """Custom inference endpoint url. None for using default public url."""
    
    client : Any
    
    model: str = "bge-large-zh-v1.5"
    
    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
                 
        self.client =  OpenAI(api_key="NO_KEY", base_url= self.api_url)
       
  
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings for a list of texts.

        Args:
            texts (Documents): A list of texts to get embeddings for.

        Returns:
            Embedded texts as List[List[float]], where each inner List[float]
                corresponds to a single input text.       
        """ 
     
        resutls = []
        response = self.client.embeddings.create(
        model=self.model,
        input=texts,
        )
        for data in response.data:            
            resutls.append(data.embedding)       
        return resutls

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
