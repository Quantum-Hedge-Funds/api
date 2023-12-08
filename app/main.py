from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.quantum_optimizer import QuantumOptimizer
from app.utils import calculate_similarity_matrix, quantum_circuit
from typing import List
from io import BytesIO
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import requests
import random

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
async def run(data: List[List[float]] = Body(), q: int | None = Body(1), algorithm: str | None = Body('classical')):
    n = len(data)
    if n == 0:
        raise HTTPException(409, {"error": "at least one asset is required"})
    if q < 0:
        raise HTTPException(400, {"error": "the required number of assets must be smaller than the total number of assets"})
    rho = calculate_similarity_matrix(data, n)
    quantum_optimizer = QuantumOptimizer(rho, n, q)
    if algorithm == 'quantum':
        result, status = quantum_optimizer.sampling_vqe_solution()
    else: 
        result, status = quantum_optimizer.exact_solution()
    values = quantum_optimizer.values
    return {"result": result, "status": status, "values": values}

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
        plt.close()
        x = [i for i in range(len(values))]
        plot = plt.plot(x,values)
        plt.xlabel('Iterations')
        plt.ylabel('Optimisation value')
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        return StreamingResponse(buffer, media_type="image/png")
    except:
      raise HTTPException(400, "Error")
    
total_jobs = 0
results = {}

def get_frequencies(data, n, q, algorithm):
    rho = calculate_similarity_matrix(data, n)
    quantum_optimizer = QuantumOptimizer(rho, n, q)
    if algorithm == 'quantum':
        result, status = quantum_optimizer.sampling_vqe_solution()
    else: 
        result, status = quantum_optimizer.exact_solution()
    values = quantum_optimizer.values
    return {"result": result, "status": status, "values": values}


@app.post("/diversify")
async def diversify(hashes: List[str] = Body(), q: int | None = Body(1), algorithm: str | None = Body('classical')):
    data = []
    for hash in hashes:
        try:
            res = requests.get(f"https://gateway.pinata.cloud/ipfs/{hash}")
            data += res.json()
        except:
            raise HTTPException(400, {"error": "invalid hash"})

    n = len(data)

    print("n", n)

    if n == 0:
        raise HTTPException(409, {"error": "alteast one asset is required"})
    if q < 0:
        raise HTTPException(400, {"error": "the required number of assets must be smaller than the total number of assets"})
    
    output = []

    status = False
    max_tries = 10
    tries = 0

    prices = []
    for i in range(len(data)):
        asset = data[i]
        asset_prices = asset.get("prices")
        asset_prices_array = [asset_prices[j].get("value") for j in range(len(asset_prices))]
        prices.append(asset_prices_array)

    while not status and tries <= max_tries:
        frequencies = get_frequencies(prices, n, q, algorithm)
        status = frequencies.get("status")
        tries += 1

    print("frequencies", frequencies)
    
    for i in range(len(data)):
        asset = data[i]
        
        output.append({
            "id": asset.get("id"),
            "frequency": frequencies.get("result")[i] * (int(random.random() * 10000)),
        })

    totalFrequency = sum([asset.get("frequency") for asset in output])

    for i in range(len(output)):
        output[i]["weight"] = int(10000 * output[i].get("frequency") / totalFrequency)

    job_id = f"job_id-{total_jobs}"
    
    results[job_id] = output
    
    return job_id

@app.post("/get-diversification-result")
async def get_diversification_results(job_id: str = Body(), q: int | None = Body(1)):
    print("results", results)
    result = results.get(job_id, None)
    
    if result == None:
        raise HTTPException(404, {"error": "job not found"})
    
    return result
