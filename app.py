import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np

# 1. Re-define the Model Class structure
class RetailTwoTower(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(RetailTwoTower, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    def forward(self, user_ids, item_ids):
        pass # Not needed for isolated inference

st.title("🛒 Two-Tower Neural Recommendation Engine")
st.markdown("Real-time candidate retrieval using a custom PyTorch architecture and FAISS.")

# 2. Load the Artifacts
@st.cache_resource
def load_artifacts():
    model = RetailTwoTower(num_users=1000, num_items=5000)
    model.load_state_dict(torch.load("two_tower_model.pth", map_location="cpu", weights_only=True))
    model.eval()
    
    index = faiss.read_index("item_vectors.faiss")
    return model, index

model, index = load_artifacts()

# 3. Interactive UI
st.sidebar.header("User Parameters")
user_id = st.sidebar.number_input("Enter User ID (0-999)", min_value=0, max_value=999, value=42)

if st.button("Generate Recommendations"):
    # Generate the user's dense embedding vector on the fly
    with torch.no_grad():
        user_tensor = torch.tensor([user_id])
        user_vector = model.user_network(model.user_embedding(user_tensor))
        user_vector = F.normalize(user_vector, p=2, dim=1).numpy()
    
    # Retrieve nearest neighbors from FAISS
    scores, indices = index.search(user_vector, k=5)
    
    st.subheader(f"Top 5 Recommendations for User #{user_id}")
    for rank, (item_idx, score) in enumerate(zip(indices[0], scores[0])):
        st.write(f"**Rank {rank+1}:** Item ID `{item_idx}` *(Similarity: {score:.3f})*")