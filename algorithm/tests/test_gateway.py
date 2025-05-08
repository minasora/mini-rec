import sys
import os
import unittest
import asyncio
import json
import numpy as np
import torch
from unittest.mock import patch, MagicMock, AsyncMock

# Add the app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../gateway-svc/app"))

# Import modules that need to be patched
os_mock = MagicMock()
# Fix the getenv mock to handle single argument calls correctly
def mock_getenv(key, default=None):
    return default
os_mock.getenv.side_effect = mock_getenv

with patch.dict('sys.modules', {
    'os': os_mock,
    'redis.asyncio': MagicMock(),
    'httpx': MagicMock(),
    'torch': torch
}):
    # Now create a mock for torch.load that will be used in main.py
    with patch('torch.load') as mock_torch_load:
        # Setup mock item embeddings
        mock_item_vec = np.array([
            [0.1, 0.1],  # Item 0
            [0.2, 0.2],  # Item 1
            [0.3, 0.3],  # Item 2
            [0.4, 0.4],  # Item 3
            [0.5, 0.5]   # Item 4
        ])
        mock_emb = MagicMock()
        mock_emb.weight = torch.nn.Parameter(torch.tensor(mock_item_vec))
        mock_torch_load.return_value = mock_emb
        
        # Now import main which will use our patched modules
        import main

class TestGateway(unittest.TestCase):
    def setUp(self):
        # Mock Redis
        self.redis_mock = AsyncMock()
        self.redis_patcher = patch('main.redis', return_value=self.redis_mock)
        self.mock_redis = self.redis_patcher.start()
        
        # Mock httpx AsyncClient
        self.httpx_client_mock = AsyncMock()
        self.httpx_client_patcher = patch('httpx.AsyncClient', return_value=self.httpx_client_mock)
        self.mock_httpx_client = self.httpx_client_patcher.start()
        
        # Test data
        self.test_user_id = 123
        self.test_k = 5
        
        # Mock ITEM_VEC
        self.original_item_vec = main.ITEM_VEC
        main.ITEM_VEC = np.array([
            [0.1, 0.1],  # Item 0
            [0.2, 0.2],  # Item 1
            [0.3, 0.3],  # Item 2
            [0.4, 0.4],  # Item 3
            [0.5, 0.5]   # Item 4
        ])
        
        # Setup event loop for async tests
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        # Restore original values
        main.ITEM_VEC = self.original_item_vec
        
        # Stop patchers
        self.redis_patcher.stop()
        self.httpx_client_patcher.stop()
        
        # Close event loop
        self.loop.close()
    
    def test_recommend(self):
        """Test the recommendation endpoint"""
        # Mock responses from services
        recall_response = MagicMock()
        recall_response.status_code = 200
        recall_response.json.return_value = [
            [1, 0.9], [2, 0.8], [3, 0.7], [4, 0.6], [5, 0.5]
        ]
        
        rank_response = MagicMock()
        rank_response.json.return_value = [
            [3, 0.95], [1, 0.85], [2, 0.75]
        ]
        
        # Configure mock client to return appropriate responses
        self.httpx_client_mock.get.return_value = recall_response
        self.httpx_client_mock.post.return_value = rank_response
        
        # Call recommend function
        result = self.loop.run_until_complete(
            main.recommend(self.test_user_id, 3)
        )
        
        # Verify result
        self.assertEqual(result, [[3, 0.95], [1, 0.85], [2, 0.75]])
        
        # Verify correct service calls were made
        self.httpx_client_mock.get.assert_called_once_with(
            f"{main.REC}/recall/{self.test_user_id}?n=150"
        )
        self.httpx_client_mock.post.assert_called_once_with(
            f"{main.RANK}/rank",
            json={"user": self.test_user_id, "items": [1, 2, 3, 4, 5], "k": 3}
        )
    
    def test_create_profile(self):
        """Test the profile creation endpoint"""
        # Test request data
        profile_request = main.ProfileReq(
            userId="test_user",
            ratings=[
                {"movieId": 1, "score": 5.0},
                {"movieId": 2, "score": 4.0},
                {"movieId": 3, "score": 4.5}
            ]
        )
        
        # Call create_profile function
        result = self.loop.run_until_complete(
            main.create_profile(profile_request)
        )
        
        # Calculate expected user vector
        expected_vec = np.array([
            main.ITEM_VEC[1] * 5.0,
            main.ITEM_VEC[2] * 4.0, 
            main.ITEM_VEC[3] * 4.5
        ]).mean(0)
        
        # Verify Redis set was called with correct parameters
        self.redis_mock.set.assert_called_once()
        call_args = self.redis_mock.set.call_args[0]
        self.assertEqual(call_args[0], "user_emb:test_user")
        saved_vec = json.loads(call_args[1])
        np.testing.assert_allclose(saved_vec, expected_vec.tolist())
        
        # Verify result
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["dim"], 2)  # Length of our test vectors
    
    def test_create_profile_empty(self):
        """Test the profile creation with invalid ratings"""
        # Test request data with invalid movie IDs
        profile_request = main.ProfileReq(
            userId="test_user",
            ratings=[
                {"movieId": 999, "score": 5.0},  # ID out of range
                {"movieId": 2, "score": 0.0},    # Score too low
                {"movieId": -1, "score": 4.5}    # Negative ID
            ]
        )
        
        # Call create_profile function
        result = self.loop.run_until_complete(
            main.create_profile(profile_request)
        )
        
        # Verify result
        self.assertEqual(result, {"error": "empty"})
        # Redis should not be called
        self.redis_mock.set.assert_not_called()

if __name__ == '__main__':
    unittest.main()
