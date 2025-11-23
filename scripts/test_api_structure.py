#!/usr/bin/env python3
"""
Simple test to verify API structure is correct.
Tests that endpoints are defined and schemas work.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all serving modules can be imported."""
    print("Testing imports...")
    
    try:
        from recsys.serving import api_schemas
        print("  ✅ api_schemas imported")
        
        from recsys.serving import ann_index
        print("  ✅ ann_index imported")
        
        from recsys.serving import explanations
        print("  ✅ explanations imported")
        
        from recsys.serving import recommender_core
        print("  ✅ recommender_core imported")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schemas():
    """Test that API schemas work."""
    print("\nTesting API schemas...")
    
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
        assert place_req.top_k == 10
        print("  ✅ PlaceRecommendationRequest works")
        
        # Test place response
        place_rec = PlaceRecommendation(
            place_id=100,
            score=0.95,
            explanations=["Test explanation"]
        )
        response = PlaceRecommendationResponse(recommendations=[place_rec])
        assert len(response.recommendations) == 1
        print("  ✅ PlaceRecommendationResponse works")
        
        # Test people request
        people_req = PeopleRecommendationRequest(
            user_id=42,
            city_id=2,
            top_k=10
        )
        assert people_req.user_id == 42
        print("  ✅ PeopleRecommendationRequest works")
        
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
        print("  ✅ PeopleRecommendationResponse works")
        
        return True
    except Exception as e:
        print(f"  ❌ Schema test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_app_structure():
    """Test that FastAPI app can be created (without loading models)."""
    print("\nTesting FastAPI app structure...")
    
    try:
        # Import without triggering startup
        from recsys.serving import api_main
        
        app = api_main.app
        assert app is not None
        print("  ✅ FastAPI app created")
        
        # Check routes
        routes = [route.path for route in app.routes]
        assert "/" in routes, "Root route missing"
        assert "/health" in routes, "Health route missing"
        assert "/recommend/places" in routes, "Places route missing"
        assert "/recommend/people" in routes, "People route missing"
        print(f"  ✅ All routes defined: {routes}")
        
        return True
    except Exception as e:
        print(f"  ❌ API structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_constants():
    """Test that constants are accessible."""
    print("\nTesting constants...")
    
    try:
        from recsys.config.constants import (
            D_MODEL,
            D_USER_RAW,
            D_PLACE_RAW,
            N_CITIES,
            C_COARSE,
            C_FINE,
            C_VIBE
        )
        
        assert D_MODEL == 128, f"D_MODEL should be 128, got {D_MODEL}"
        assert D_USER_RAW == 148, f"D_USER_RAW should be 148, got {D_USER_RAW}"
        assert D_PLACE_RAW == 114, f"D_PLACE_RAW should be 114, got {D_PLACE_RAW}"
        assert N_CITIES == 8, f"N_CITIES should be 8, got {N_CITIES}"
        
        print(f"  ✅ Constants validated:")
        print(f"     D_MODEL={D_MODEL}, D_USER_RAW={D_USER_RAW}, D_PLACE_RAW={D_PLACE_RAW}")
        print(f"     N_CITIES={N_CITIES}, C_COARSE={C_COARSE}, C_FINE={C_FINE}, C_VIBE={C_VIBE}")
        
        return True
    except Exception as e:
        print(f"  ❌ Constants test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all structure tests."""
    print("="*60)
    print("Team 2: API Structure Validation")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Constants", test_constants()))
    results.append(("API Schemas", test_schemas()))
    results.append(("FastAPI App", test_api_app_structure()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All structure tests passed!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Wait for Team 1's deliverables (embeddings, model checkpoint)")
        print("  3. Run: python scripts/validate_team1_deliverables.py")
        print("  4. Run: python scripts/run_build_indices.py")
        print("  5. Start server: uvicorn recsys.serving.api_main:app --port 8000")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

