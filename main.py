from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from quantum_optimizer import QuantumOptimizer
from utils import calculate_similarity_matrix
from typing import List

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run")
async def run(data: List[List[float]] = Body(), q: int | None = Body(1)):
    n = len(data)
    rho = calculate_similarity_matrix(data, n)
    quantum_optimizer = QuantumOptimizer(rho, n, q)
    svqe_state, svqe_level = quantum_optimizer.sampling_vqe_solution()
    return {"result": list(svqe_state)}