from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from quantum_optimizer import QuantumOptimizer
from utils import calculate_similarity_matrix
from typing import List
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

total_jobs = 0

results = {}

@app.post("/diversify")
async def diversify(hashes: List[str] = Body(), q: int | None = Body(1)):
    data = []
    for hash in hashes:
        try:
            res = requests.get(f"https://gateway.pinata.cloud/ipfs/{hash}")
            data += res.json()
        except:
            raise HTTPException(400, {"error": "invalid hash"})

    n = len(data)
    if n == 0:
        raise HTTPException(409, {"error": "alteast one asset is required"})
    q_fix = n-q
    if q_fix < 0:
        raise HTTPException(400, {"error": "the required number of assets must be smaller than the total number of assets"})
    

    output = []
    
    for i in range(len(data)):
        asset = data[i]
        
        output.append({
            "id": asset.get("id"),
            "frequecy": int(random.random() * 10000),
        })

    totalFrequency = sum([asset.get("frequecy") for asset in output])

    for i in range(len(output)):
        output[i]["weight"] = int(10000 * output[i].get("frequecy") / totalFrequency)

    job_id = f"job_id-{total_jobs}"
    
    results[job_id] = output
    
    return job_id

@app.post("/get-diversification-result")
async def get_diversification_results(job_id: str = Body()):
    result = results.get(job_id, None)
    
    if result == None:
        raise HTTPException(404, {"error": "job not found"})
    
    return result