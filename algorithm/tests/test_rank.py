import sys
import os
import unittest
import asyncio
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Add the app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../ranker-svc/app"))

# Import settings and patch before importing rank
import settings

# Mock the torch.jit.load and torch.load functions
with patch('torch.jit.load') as mock_jit_load, \
     patch('torch.load') as mock_load:
    
    # Setup mock returns
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_jit_load.return_value = mock_model
    
    # Mock user embeddings
    mock_user_emb = MagicMock()
    mock_user_emb.weight = torch.nn.Parameter(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
    
    # Mock item embeddings
    mock_item_emb = MagicMock()
    mock_item_emb.weight = torch.nn.Parameter(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]))
    
    mock_load.side_effect = [mock_user_emb, mock_item_emb]
    
    # Now import rank which will use our patched functions
    import rank

class TestRank(unittest.TestCase):
    def setUp(self):
        # Mock the external dependencies
        self.redis_mock = MagicMock()
        self.redis_patcher = patch('rank._redis', return_value=self.redis_mock)
        self.mock_redis = self.redis_patcher.start()
        
        # Create test data
        self.test_user_id = 1
        self.test_items = [1, 2, 3, 4, 5]
        self.test_k = 3
        
        # Ensure we have a test loop for async tests
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def tearDown(self):
        self.redis_patcher.stop()
        self.loop.close()
    
    def test_user_vec_from_embedding(self):
        """Test retrieving user vector from embedding matrix"""
        # Save original UE
        original_ue = rank.UE
        
        # Replace with test data
        rank.UE = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        
        # Test for existing user
        result = self.loop.run_until_complete(rank.user_vec(1))
        np.testing.assert_array_equal(result, np.array([0.3, 0.4]))
        
        # Restore original UE
        rank.UE = original_ue
    
    def test_user_vec_from_redis(self):
        """Test retrieving user vector from Redis"""
        # Mock Redis response with async function
        async def mock_get(key):
            if key == 'user_emb:test_user':
                return '[[0.7, 0.8]]'
            return None
            
        self.redis_mock.get = mock_get
        
        # Test for cached user
        result = self.loop.run_until_complete(rank.user_vec('test_user'))
        np.testing.assert_array_equal(result, np.array([[0.7, 0.8]], dtype='float32'))
    
    def test_rank_items(self):
        """Test ranking items for a user"""
        # We don't need to reset original values since we patched them globally
        
        # Fix the tensor dimensions to work with torch.cat
        # Override the user_vec function to return a compatible tensor
        async def mock_user_vec(user_id):
            return np.array([0.3, 0.4])  # Same dim as original UE[1]
            
        with patch.object(rank, 'user_vec', side_effect=mock_user_vec), \
             patch.object(rank.MODEL, '__call__', return_value=torch.tensor([[0.2], [0.5], [0.1], [0.9], [0.3]])):
            # Run test
            result = self.loop.run_until_complete(rank.rank(1, [0, 1, 2, 3, 4], 3))
        
        # We expect items 3, 1, 4 to be top 3 with scores 0.9, 0.5, 0.3
        expected = [(3, 0.9), (1, 0.5), (4, 0.3)]
        
        # Verify results match expected top items and scores
        for i, (item_id, score) in enumerate(result):
            self.assertEqual(item_id, expected[i][0])
            self.assertAlmostEqual(score, expected[i][1])

if __name__ == '__main__':
    unittest.main()
