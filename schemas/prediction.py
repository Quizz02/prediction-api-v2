from typing import Optional
from pydantic import BaseModel

class Prediction(BaseModel):
    id : int
    animalType : str
    size : str
    color : str
    district : str
    status : Optional[str]
    p_animal : Optional[int]
    p_distrito : Optional[int]
    p_size : Optional[int]
    p_color : Optional[int]