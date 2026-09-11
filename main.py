import os
from typing import List, Optional

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestErrorModel
import onnxruntime as ort
from pydantic import BaseModel, HttpUrl
from torch_geometric import data

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
STATIC_DIR = os.path.join(BASE_DIR, "Static")

model_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

app = FastAPI()


@app.get("/")
def home_page():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "service": "Simple GCN Cora API",
        "status": "Running",
    }


class GraphPredictRequest(BaseModel):
    node_feature: List[List[float]]
    edge_index: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    node_index: List[int]


def softmax(scores: np.ndarray):
    shifted = scores - scores.max(keepdims=True, axis=1)
    expo = np.exp(shifted)
    return expo / expo.sum(axis=-1, keepdims=True)


def run_model(node_features: np.ndarray, edge_index: np.ndarray, node_index_to_return):
    outputs = model_session.run(
        ["logits"],
        {
            "node_features": node_features.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
        },
    )
    logits = outputs[0]
    probs = softmax(logits)
    predicted_class = logits.argmax(axis=-1)

    results = []
    for i in node_index_to_return:
        results.append(
            {
                "node_index": i,
                "predicted_class_id": int(predicted_class[i]),
                "predicted_class_name": CORA_CLASSES[int(predicted_class[i])],
                "probability": probs[i].tolist(),
                "logits": logits[i].tolist(),
            }
        )

    return {
        "num_nodes": node_features.shape[0],
        "num_edges": edge_index.shape[1],
        "predictions": results,
    }


@app.get("/health")
def health_check():
    return {
        "status": "Healthy",
        "providers": model_session.get_providers(),
    }


@app.get("/model_info")
def model_info():
    return {
        "model_name": "SimpleGCN",
        "feature_dimension": 1433,
        "num_classes": 7,
        "class_mapping": CORA_CLASSES,
        "inputs": [
            {
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            }
            for inp in model_session.get_inputs()
        ],
        "outputs": [
            {
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            }
            for out in model_session.get_outputs()
        ],
    }


@app.post("/predict")
def predict_custom_graph(request: GraphPredictRequest):
    if not request.node_feature:
        raise HTTPException(400, "Node Feature cannot be empty")

    for feat_vec in request.node_feature:
        if len(feat_vec) != 1433:
            raise HTTPException(
                422, "Each node's feature vector must exactly has 1433 features"
            )

    node_features = np.array(request.node_feature, dtype=np.float32)
    num_nodes = len(request.node_feature)

    if request.edge_index:
        edge_index = np.array(request.edge_index, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise HTTPException(422, "Edge indices must have shape [2, num_edges]")
    else:
        node_idx = np.arange(num_nodes, dtype=np.int64)  # Self loop
        edge_index = np.vstack([node_idx, node_idx])

    all_node_features = list(range(num_nodes))

    return run_model(node_features, edge_index, all_node_features)


@app.post("/predict/cora_node")
def predict_real_cora_nodes(request: CoraNodeRequest):
    from torch_geometric.datasets import Planetoid

    try:
        cora_dataset = Planetoid(root=DATA_DIR, name="Cora")[0]
    except Exception as error:
        raise HTTPException(500, f"Failed to load cora dataset\nError: {error}")

    largest_valid_idx = cora_dataset.num_nodes - 1

    invalid_idx = [
        idx for idx in request.node_index if idx < 0 or idx > largest_valid_idx
    ]

    if invalid_idx:
        raise HTTPException(
            400, f"Node index out of bound (must be between 0 to {largest_valid_idx})"
        )

    return run_model(
        cora_dataset.x.numpy(), cora_dataset.edge_index.numpy(), request.node_index
    )


if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
