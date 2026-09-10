import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

CORA_CLASSES = {
    0: "Case_Based",
    1: "Genetic_Algorithms",
    2: "Neural_Networks",
    3: "Probabilistic_Methods",
    4: "Reinforcement_Learning",
    5: "Rule_Learning",
    6: "Theory",
}

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "simple_gcn_cora.onnx")
DATA_DIR = os.path.join(BASE_DIR, "data", "Planetoid")


app = FastAPI()


class GraphPredictRequest(BaseModel):
    node_feature: List[List[float]]
    edge_index: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    node_index: List[int]


