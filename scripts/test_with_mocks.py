#!/usr/bin/env python3
"""
Test the serving components with mock data.
Use this during independent development (Days 1-9).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_mock_embeddings(output_dir: str = "test_data"):
    """Create mock embeddings for testing."""
    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/embeddings").mkdir(exist_ok=True)
    
    print("Creating mock embeddings...")
    
    # Mock user embeddings
    n_users = 100
    user_ids = list(range(n_users))
    user_embeddings = [np.random.randn(128).tolist() for _ in range(n_users)]
    user_df = pd.DataFrame({
        'user_id': user_ids,
        'embedding': user_embeddings
    })
    user_df.to_parquet(f'{output_dir}/embeddings/user_embeddings.parquet')
    print(f"  ✅ Created {n_users} mock user embeddings")
    
    # Mock place embeddings
    n_places = 200
    place_ids = list(range(n_places))
    place_embeddings = [np.random.randn(128).tolist() for _ in range(n_places)]
    place_df = pd.DataFrame({
        'place_id': place_ids,
        'embedding': place_embeddings
    })
    place_df.to_parquet(f'{output_dir}/embeddings/place_embeddings.parquet')
    print(f"  ✅ Created {n_places} mock place embeddings")
    
    return f'{output_dir}/embeddings'


def test_embedding_store():
    """Test EmbeddingStore with mock data."""
    print("\n" + "="*50)
    print("Testing EmbeddingStore...")
    
    embeddings_dir = create_mock_embeddings()
    
    try:
        from recsys.serving.recommender_core import EmbeddingStore
        
        store = EmbeddingStore()
        store.load_from_parquet(
            f'{embeddings_dir}/user_embeddings.parquet',
            f'{embeddings_dir}/place_embeddings.parquet'
        )
        
        z_u = store.get_user_embedding(42)
        assert z_u is not None, "Failed to retrieve user embedding"
        assert z_u.shape == (128,), f"Wrong shape: {z_u.shape}, expected (128,)"
        print(f"  ✅ Retrieved user embedding: shape {z_u.shape}")
        
        z_p = store.get_place_embedding(100)
        assert z_p is not None, "Failed to retrieve place embedding"
        assert z_p.shape == (128,), f"Wrong shape: {z_p.shape}, expected (128,)"
        print(f"  ✅ Retrieved place embedding: shape {z_p.shape}")
        
        print("✅ EmbeddingStore test passed!")
        return True
    except Exception as e:
        print(f"❌ EmbeddingStore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ann_index():
    """Test ANN index with mock data."""
    print("\n" + "="*50)
    print("Testing ANN Index...")
    
    try:
        from recsys.serving.ann_index import AnnIndex
        
        # Create dummy embeddings
        n_items = 100
        dummy_embeddings = np.random.randn(n_items, 128).astype(np.float32)
        dummy_ids = list(range(n_items))
        
        # Build index
        index = AnnIndex(dimension=128, metric='cosine')
        index.build(dummy_embeddings, dummy_ids)
        print(f"  ✅ Built index with {n_items} items")
        
        # Test search
        query = np.random.randn(128).astype(np.float32)
        results = index.search(query, top_k=10)
        
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results), "Invalid result format"
        print(f"  ✅ Search returned {len(results)} results")
        
        print("✅ ANN Index test passed!")
        return True
    except Exception as e:
        print(f"❌ ANN Index test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_city_ann_manager():
    """Test CityAnnIndexManager with mock data."""
    print("\n" + "="*50)
    print("Testing CityAnnIndexManager...")
    
    try:
        from recsys.serving.ann_index import CityAnnIndexManager
        
        manager = CityAnnIndexManager(dimension=128)
        
        # Create mock data for 2 cities
        for city_id in [0, 1]:
            n_items = 50
            embeddings = np.random.randn(n_items, 128).astype(np.float32)
            ids = list(range(city_id * 100, city_id * 100 + n_items))
            
            manager.build_city_index(city_id, embeddings, ids)
            print(f"  ✅ Built index for city {city_id} with {n_items} items")
        
        # Test search
        query = np.random.randn(128).astype(np.float32)
        results = manager.search(0, query, top_k=5)
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        print(f"  ✅ Search in city 0 returned {len(results)} results")
        
        # Test save/load
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            manager.save(tmpdir, "test")
            print(f"  ✅ Saved indices to {tmpdir}")
            
            # Load into new manager
            new_manager = CityAnnIndexManager(dimension=128)
            new_manager.load(tmpdir, "test", [0, 1])
            results2 = new_manager.search(0, query, top_k=5)
            assert len(results2) == 5, "Failed to load index"
            print(f"  ✅ Loaded indices and verified search works")
        
        print("✅ CityAnnIndexManager test passed!")
        return True
    except Exception as e:
        print(f"❌ CityAnnIndexManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_schemas():
    """Test API schemas."""
    print("\n" + "="*50)
    print("Testing API Schemas...")
    
    try:
        from recsys.serving.api_schemas import (
            PlaceRecommendationRequest,
            PlaceRecommendationResponse,
            PlaceRecommendation,
            PeopleRecommendationRequest,
            PeopleRecommendationResponse,
            PeopleRecommendation
        )
        
        # Test place request
        place_req = PlaceRecommendationRequest(
            user_id=42,
            city_id=2,
            time_slot=3,
            desired_categories=[0, 2],
            top_k=10
        )
        assert place_req.user_id == 42
        print("  ✅ PlaceRecommendationRequest created")
        
        # Test place response
        place_rec = PlaceRecommendation(
            place_id=100,
            score=0.95,
            explanations=["Test explanation"]
        )
        response = PlaceRecommendationResponse(recommendations=[place_rec])
        assert len(response.recommendations) == 1
        print("  ✅ PlaceRecommendationResponse created")
        
        # Test people request
        people_req = PeopleRecommendationRequest(
            user_id=42,
            city_id=2,
            top_k=10
        )
        assert people_req.user_id == 42
        print("  ✅ PeopleRecommendationRequest created")
        
        # Test people response
        people_rec = PeopleRecommendation(
            user_id=789,
            compat_score=0.82,
            attend_prob=0.75,
            combined_score=0.799,
            explanations=["Test explanation"]
        )
        response = PeopleRecommendationResponse(recommendations=[people_rec])
        assert len(response.recommendations) == 1
        print("  ✅ PeopleRecommendationResponse created")
        
        print("✅ API Schemas test passed!")
        return True
    except Exception as e:
        print(f"❌ API Schemas test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*50)
    print("Testing Serving Components with Mock Data")
    print("="*50)
    
    results = []
    
    results.append(("API Schemas", test_api_schemas()))
    results.append(("ANN Index", test_ann_index()))
    results.append(("City ANN Manager", test_city_ann_manager()))
    results.append(("EmbeddingStore", test_embedding_store()))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for integration with Team 1's deliverables.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before integration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

