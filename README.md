# 🎬 Movie Recommendation System

This project implements a complete end-to-end **movie recommendation system** using two collaborative filtering approaches:

- **Item–Item Collaborative Filtering** using cosine similarity
- **Matrix Factorization (FunkMF)** trained via stochastic gradient descent

The system analyzes user–movie rating patterns to recommend similar movies and generate personalized recommendations for individual users.

To highlight different aspects of building and deploying ML-powered applications, the project includes:

- **Offline artifact building**
  - A reproducible script that prepares data, computes item–item similarity matrices, and trains a matrix factorization model
  - All models and intermediate results are saved as reusable artifacts

- **Gradio web interface**
  - An interactive UI for exploring similar movies and user-specific recommendations
  - Supports both item–item collaborative filtering and FunkMF-based recommendations in real time

- **Dockerized deployment**
  - A containerized setup for running the Gradio app consistently across environments
  - Enables easy local or remote deployment without manual environment configuration

Together, these components demonstrate how recommendation models can be trained offline, served efficiently for real-time inference, and exposed through a modern, user-friendly interface.

-----

## Instructions (Run Locally)

### 1) Clone the repo

The first step is to clone the repo, and then go the directory.

```
git clone https://github.com/Sam-Gartenstein/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2) Create and activate a virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3) Install dependencies

```
pip install -r requirements.txt
```

### 4) Build artifacts (one-time setup)

This generates the saved `.pkl` files used by the app, including the trained FunkMF model:

```
PYTHONPATH=./src python -m recommender.build_artifacts
```

### 5) Run the Gradio app

```
PYTHONPATH=./src python -m app.gradio_app
```

**Open**:
- http://localhost:7860/

### 6) Stop and deactivate

Stop the app with:

```
Ctrl + C
```

Deactivate the environment:

```
deactivate
```
