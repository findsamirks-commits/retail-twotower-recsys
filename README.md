# Retail Two-Tower Recommendation Engine

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?logo=streamlit)](https://retail-twotower-recsys-m5umimd3ge9avnvkqojwzv.streamlit.app/)

## Description
This repository hosts a custom **PyTorch Two-Tower Recommendation System** built for modern retail platforms. The architecture utilizes dual neural networks (towers) to independently map users and items into a shared latent vector space. By integrating **FAISS (Facebook AI Similarity Search)**, the engine performs lightning-fast vector similarity retrieval to generate highly relevant, personalized product recommendations in real-time. The end-to-end pipeline is deployed via an interactive **Streamlit** front-end for seamless inference.

🌐 **Live Application:** [View the Streamlit App Here](https://retail-twotower-recsys-m5umimd3ge9avnvkqojwzv.streamlit.app/)

## Topics Covered
- `PyTorch`
- `Two-Tower Neural Networks`
- `Recommender Systems`
- `FAISS` (Vector Search)
- `Retail Analytics & Personalization`
- `Streamlit Deployment`
- `Deep Learning`

## Repository Structure
- **`app.py`**: The main Streamlit application file handling the web interface and model inference.
- **`two_tower_model.pth`**: The trained PyTorch model weights for the user and item embedding towers.
- **`item_vectors.faiss`**: The exported FAISS index containing pre-computed item vectors for optimized nearest-neighbor retrieval.
- **`requirements.txt`**: The necessary Python dependencies required to execute the application.

## About the Author
This project is part of a broader suite of **Retail Agentic AI & Data Science** architectures designed to bridge the gap between high-volume retail operations and autonomous machine learning systems.
