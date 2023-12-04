from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from quantum_optimizer import QuantumOptimizer
from utils import calculate_similarity_matrix, quantum_circuit
from typing import List
from io import BytesIO
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt

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
    result = quantum_optimizer.sampling_vqe_solution()
    values = quantum_optimizer.values
    return {"result": result, "values": values}

@app.post("/draw")
async def draw(n: int | None = Body(2), q: int | None = Body(1)):
    try:
      circuit = quantum_circuit(n)
      drawn_image = circuit.draw(output="mpl")
      buffer = BytesIO()
      drawn_image.savefig(buffer, format="png")
      buffer.seek(0)
      return StreamingResponse(buffer, media_type="image/png")
    except:
      raise HTTPException(400, "Invalid Circuit")
    
@app.post("/plot")
async def plot(values: List[float] = Body(), q: int | None = Body(1)):
    try:
        x = [i for i in range(len(values))]
        plot = plt.plot(x,values)
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")
    except:
      raise HTTPException(400, "Error")
