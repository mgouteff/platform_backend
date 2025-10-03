from pydantic import BaseModel

class QueryRequest(BaseModel):
    """
    Defines the request body for the /query endpoint.
    - question: the natural language query from the user
    - top_k: how many results to return (default = 3)
    """
    question: str
    top_k: int = 10

