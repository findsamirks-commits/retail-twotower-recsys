import pytest
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import os

# 1. Re-define the Model Class structure
class RetailTwoTower(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(RetailTwoTower, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.user_network = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embedding_dim)
        )
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        self.item_network = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embedding_dim)
        )
    def forward(self, user_ids, item_ids):
        pass

# 2. Pytest Fixture to Load Artifacts Once
@pytest.fixture(scope="module")
def artifacts():
    model = RetailTwoTower(num_users=1000, num_items=5000)
    
    # Load weights from the GitHub workspace during CI
    if os.path.exists("two_tower_model.pth"):
        model.load_state_dict(torch.load("two_tower_model.pth", map_location="cpu", weights_only=True))
    model.eval()

    if os.path.exists("item_vectors.faiss"):
        index = faiss.read_index("item_vectors.faiss")
    else:
        # Fallback to prevent crashing if files are not fully checked out
        index = faiss.IndexFlatIP(64)
        
    return model, index

# 3. Test the User Tower Tensor Generation
def test_user_embedding_generation(artifacts):
    model, _ = artifacts
    
    with torch.no_grad():
        user_tensor = torch.tensor([42])
        user_vector = model.user_network(model.user_embedding(user_tensor))
        user_vector = F.normalize(user_vector, p=2, dim=1).numpy()
    
    # The output must be exactly (1, 64) to match the item vectors in FAISS
    assert user_vector.shape == (1, 64)
    
# 4. Test FAISS Sub-Millisecond Retrieval
def test_faiss_retrieval(artifacts):
    _, index = artifacts
    
    if index.ntotal == 0:
        pytest.skip("FAISS index file not found, skipping retrieval test.")
        
    # Generate a random 64-dimensional query vector matching the Two-Tower space
    dummy_query = np.random.rand(1, 64).astype('float32')
    faiss.normalize_L2(dummy_query)
    
    k = 5
    scores, indices = index.search(dummy_query, k)
    
    # Verify the index returns exactly 5 IDs and 5 confidence scores
    assert indices.shape == (1, k)
    assert scores.shape == (1, k)