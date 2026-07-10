# Hybrid AI Router

## Overview

Hybrid AI Router is an intelligent routing system developed for the AMD Developer Hackathon.

It automatically analyzes each incoming task and decides whether it should be processed by a **local Large Language Model (0 cloud tokens)** or a **Fireworks AI hosted model**.

The primary goal is to minimize cloud token usage while maintaining high response quality and low latency.

---

## Features

- Automatic task classification
- Prompt complexity estimation
- Hybrid routing (Local + Fireworks)
- Multiple Fireworks model selection
- Local LLM inference server
- FastAPI-based local API
- Token usage tracking
- Latency tracking
- Docker-ready architecture

---

## Project Structure

```text
Hybrid-Router/
│
├── agent.py
├── router.py
├── config.py
├── fireworks_client.py
├── local_client.py
├── stats.py
├── requirements.txt
├── README.md
│
├── input/
├── output/
├── models/
│
└── local_server/
    ├── server.py
    ├── model_loader.py
    ├── inference.py
    ├── download_model.py
    └── config.py
```

---

## Routing Strategy

The router performs three stages:

1. NLP task classification
2. Prompt complexity estimation
3. Intelligent model selection

Simple tasks are routed to the local model whenever possible to eliminate cloud token usage.

More complex tasks such as:

- Code Generation
- Code Debugging
- Logical Reasoning
- Mathematical Reasoning

are routed to Fireworks AI for higher accuracy.

---

## Environment Variables

The following environment variables are required:

```env
FIREWORKS_API_KEY=<your_api_key>

FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1

ALLOWED_MODELS=model1,model2
```

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Local Model Server

```bash
python local_server/server.py
```

---

## Running the Agent

```bash
python agent.py
```

---

## Docker

Build the Docker image:

```bash
docker build -t hybrid-router .
```

Run the container:

```bash
docker run hybrid-router
```

---

## Technologies Used

- Python
- FastAPI
- Hugging Face Transformers
- Fireworks AI
- PyTorch
- Accelerate