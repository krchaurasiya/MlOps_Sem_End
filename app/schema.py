from pydantic import BaseModel

class InputData(BaseModel):
    rating: float
    review_count: int
    category: str
    text: str   # user input (description/search)