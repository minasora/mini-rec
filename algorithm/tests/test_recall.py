import sys
import os
import unittest
import asyncio
import numpy as np
import json
import faiss
import torch
from unittest.mock import patch, MagicMock, AsyncMock

# Add the app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../rec-user-svc/app"))

# Import settings and patch loading functionality before importing recall
import settings

# First, let's patch the faiss and torch loading functions
with patch('faiss.read_index') as mock_read_index, \
     patch('numpy.load') as mock_np_load, \
     patch('torch.load') as mock_torch_load:
     
    # Create mock FAISS index
    mock_index = MagicMock()
    mock_index.search.return_value = (
        np.array([[0.95, 0.85, 0.75, 0.65, 0.55]]),  # distances
        np.array([[1, 2, 3, 4, 5]])                  # indices
    )
    mock_read_index.return_value = mock_index
    
    # Create mock movie IDs
    mock_movie_ids = np.array([100, 101, 102, 103, 104, 105, 106])
    mock_np_load.return_value = mock_movie_ids
    
    # Create mock user embeddings
    mock_user_emb = MagicMock()
    mock_user_emb.weight = torch.nn.Parameter(torch.randn(10, 64)) # 10 users, 64-dim embeddings
    mock_torch_load.return_value = mock_user_emb
    
    # Now import recall which will use our patched functions
    import recall

class TestRecall(unittest.TestCase):
    def setUp(self):
        # Mock Redis
        self.redis_mock = AsyncMock()
        self.redis_patcher = patch('recall.redis', return_value=self.redis_mock)
        self.mock_redis = self.redis_patcher.start()
        
        # Mock FAISS index and other globals
        self.original_index = recall.INDEX
        self.original_item_id = recall.ITEM_ID
        self.original_user_emb = recall.USER_EMB
        
        # Setup test data
        self.test_user_id = 1
        self.test_topn = 5
        
        # Create test loop for async tests
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def tearDown(self):
        # Restore original values
        recall.INDEX = self.original_index
        recall.ITEM_ID = self.original_item_id
        recall.USER_EMB = self.original_user_emb
        
        # Stop patchers
        self.redis_patcher.stop()
        
        # Close event loop
        self.loop.close()
    
    def test_recall_from_cache(self):
        """Test retrieving recall results from Redis cache"""
        # Mock Redis to return cached results - using lists instead of tuples to match JSON format
        cached_results = [[101, 0.95], [102, 0.85], [103, 0.75]]
        self.redis_mock.exists.return_value = True
        self.redis_mock.get.return_value = json.dumps(cached_results)
        
        # Call recall function
        result = self.loop.run_until_complete(
            recall.recall(self.test_user_id, self.test_topn)
        )
        
        # Verify results - the function should return data in the same format as cached
        self.assertEqual(result, cached_results)
        self.redis_mock.exists.assert_called_once_with(f"recall:{self.test_user_id}:{self.test_topn}")
        self.redis_mock.get.assert_called_once_with(f"recall:{self.test_user_id}:{self.test_topn}")
    
    def test_recall_from_index(self):
        """Test retrieving recall results from FAISS index"""
        # Mock Redis to indicate cache miss
        self.redis_mock.exists.return_value = False
        
        # Mock FAISS search results
        mock_index = MagicMock()
        mock_distances = np.array([[0.95, 0.85, 0.75, 0.65, 0.55]])
        mock_indices = np.array([[1, 2, 3, 4, 5]])
        mock_index.search.return_value = (mock_distances, mock_indices)
        recall.INDEX = mock_index
        
        # Mock item IDs
        recall.ITEM_ID = np.array([100, 101, 102, 103, 104, 105, 106])
        
        # Mock user embeddings
        recall.USER_EMB = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ]).astype('float32')
        
        # Call recall function
        result = self.loop.run_until_complete(
            recall.recall(self.test_user_id, self.test_topn)
        )
        
        # Expected results based on mocks
        expected = [
            (101, 0.95),
            (102, 0.85),
            (103, 0.75),
            (104, 0.65),
            (105, 0.55)
        ]
        
        # Verify results
        self.assertEqual(len(result), len(expected))
        for i in range(len(result)):
            self.assertEqual(result[i][0], expected[i][0])
            self.assertAlmostEqual(result[i][1], expected[i][1])
        
        # Verify Redis set was called to cache results
        self.redis_mock.set.assert_called_once()

if __name__ == '__main__':
    unittest.main()
