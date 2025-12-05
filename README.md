# 🎬 Movie Recommendation System

This project implements a complete end-to-end **movie recommendation system** using **item–item collaborative filtering**.  
The system analyzes user–movie interactions to compute similarity scores between items and generate personalized movie recommendations.

To highlight different aspects of building ML-powered applications, the project includes:

- **FastAPI backend**  
  Exposes API endpoints (`/similar`, `/recommend`) with typed responses and a service health check.

- **Dockerized deployment**  
  Provides a reproducible environment for running the recommender and serving the API.

- **Gradio web interface**  
  Enables interactive exploration of similar movies and user-specific recommendations in a simple UI.

Together, these components demonstrate how recommendation models can be packaged, served, and surfaced through modern production tools.

---

## 🚧 Next Steps

Planned enhancements include:

- Adding **matrix factorization models** (e.g., SVD)  
- Experimenting with hybrid recommendation strategies  
