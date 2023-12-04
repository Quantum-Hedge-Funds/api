from fastapi import FastAPI, Body, HTTPException
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
    if n == 0:
        raise HTTPException(409, {"error": "alteast one asset is required"})
    q_fix = n-q
    if q_fix < 0:
        raise HTTPException(400, {"error": "the required number of assets must be smaller than the total number of assets"})
    rho = calculate_similarity_matrix(data, n)
    quantum_optimizer = QuantumOptimizer(rho, n, q_fix)
    svqe_state, svqe_level = quantum_optimizer.sampling_vqe_solution()
    print(svqe_state)
    return {"result": list(svqe_state[-n:])}