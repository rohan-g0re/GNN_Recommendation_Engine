#!/usr/bin/env python3
"""
Load testing script for the recommendation API.
Tests both /recommend/places and /recommend/people endpoints.
"""

import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import argparse
import json


class LoadTester:
    """Load tester for recommendation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def test_health(self) -> bool:
        """Test health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def recommend_places(self, user_id: int, city_id: int = None, **kwargs) -> Dict:
        """Make a place recommendation request."""
        payload = {
            "user_id": user_id,
            "top_k": kwargs.get("top_k", 10)
        }
        if city_id is not None:
            payload["city_id"] = city_id
        if "time_slot" in kwargs:
            payload["time_slot"] = kwargs["time_slot"]
        if "desired_categories" in kwargs:
            payload["desired_categories"] = kwargs["desired_categories"]
        
        response = requests.post(
            f"{self.base_url}/recommend/places",
            json=payload,
            timeout=30
        )
        return {
            "status_code": response.status_code,
            "response_time": response.elapsed.total_seconds(),
            "success": response.status_code == 200,
            "data": response.json() if response.status_code == 200 else None
        }
    
    def recommend_people(self, user_id: int, city_id: int = None, **kwargs) -> Dict:
        """Make a people recommendation request."""
        payload = {
            "user_id": user_id,
            "top_k": kwargs.get("top_k", 10)
        }
        if city_id is not None:
            payload["city_id"] = city_id
        if "target_place_id" in kwargs:
            payload["target_place_id"] = kwargs["target_place_id"]
        if "activity_tags" in kwargs:
            payload["activity_tags"] = kwargs["activity_tags"]
        
        response = requests.post(
            f"{self.base_url}/recommend/people",
            json=payload,
            timeout=30
        )
        return {
            "status_code": response.status_code,
            "response_time": response.elapsed.total_seconds(),
            "success": response.status_code == 200,
            "data": response.json() if response.status_code == 200 else None
        }
    
    def run_single_request(self, endpoint: str, user_id: int, **kwargs) -> Dict:
        """Run a single request."""
        if endpoint == "places":
            return self.recommend_places(user_id, **kwargs)
        elif endpoint == "people":
            return self.recommend_people(user_id, **kwargs)
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
    
    def run_load_test(
        self,
        endpoint: str,
        num_requests: int,
        concurrent: int = 10,
        user_ids: List[int] = None,
        **kwargs
    ) -> Dict:
        """
        Run load test.
        
        Args:
            endpoint: "places" or "people"
            num_requests: Total number of requests
            concurrent: Number of concurrent requests
            user_ids: List of user IDs to use (will cycle if shorter than num_requests)
            **kwargs: Additional parameters for requests
        """
        if user_ids is None:
            user_ids = list(range(1, num_requests + 1))
        
        # Cycle user_ids if needed
        if len(user_ids) < num_requests:
            user_ids = (user_ids * ((num_requests // len(user_ids)) + 1))[:num_requests]
        
        results = []
        start_time = time.time()
        
        print(f"\nRunning load test: {num_requests} requests, {concurrent} concurrent")
        print(f"Endpoint: /recommend/{endpoint}")
        
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = []
            for i in range(num_requests):
                user_id = user_ids[i % len(user_ids)]
                future = executor.submit(self.run_single_request, endpoint, user_id, **kwargs)
                futures.append(future)
            
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % 10 == 0:
                        print(f"  Completed: {completed}/{num_requests}")
                except Exception as e:
                    results.append({
                        "status_code": 0,
                        "response_time": 0,
                        "success": False,
                        "error": str(e)
                    })
                    completed += 1
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        response_times = [r["response_time"] for r in results if r.get("response_time", 0) > 0]
        success_count = sum(1 for r in results if r.get("success", False))
        error_count = num_requests - success_count
        
        stats = {
            "total_requests": num_requests,
            "successful_requests": success_count,
            "failed_requests": error_count,
            "success_rate": success_count / num_requests if num_requests > 0 else 0,
            "total_time_seconds": total_time,
            "requests_per_second": num_requests / total_time if total_time > 0 else 0,
            "response_times": {
                "mean": statistics.mean(response_times) if response_times else 0,
                "median": statistics.median(response_times) if response_times else 0,
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "p95": sorted(response_times)[int(len(response_times) * 0.95)] if len(response_times) > 0 else 0,
                "p99": sorted(response_times)[int(len(response_times) * 0.99)] if len(response_times) > 0 else 0,
            } if response_times else {}
        }
        
        return {
            "endpoint": endpoint,
            "stats": stats,
            "results": results
        }
    
    def print_results(self, test_results: Dict):
        """Print formatted test results."""
        stats = test_results["stats"]
        
        print("\n" + "="*60)
        print(f"Load Test Results: /recommend/{test_results['endpoint']}")
        print("="*60)
        print(f"Total Requests:     {stats['total_requests']}")
        print(f"Successful:         {stats['successful_requests']}")
        print(f"Failed:             {stats['failed_requests']}")
        print(f"Success Rate:       {stats['success_rate']:.2%}")
        print(f"\nTotal Time:         {stats['total_time_seconds']:.2f}s")
        print(f"Requests/Second:    {stats['requests_per_second']:.2f}")
        
        if stats['response_times']:
            rt = stats['response_times']
            print(f"\nResponse Times:")
            print(f"  Mean:             {rt['mean']:.3f}s")
            print(f"  Median:           {rt['median']:.3f}s")
            print(f"  Min:              {rt['min']:.3f}s")
            print(f"  Max:              {rt['max']:.3f}s")
            print(f"  P95:              {rt['p95']:.3f}s")
            print(f"  P99:              {rt['p99']:.3f}s")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Load test the recommendation API')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                        help='Base URL of the API')
    parser.add_argument('--endpoint', type=str, choices=['places', 'people', 'both'],
                        default='both', help='Endpoint to test')
    parser.add_argument('--requests', type=int, default=100,
                        help='Total number of requests')
    parser.add_argument('--concurrent', type=int, default=10,
                        help='Number of concurrent requests')
    parser.add_argument('--user-ids', type=str, default=None,
                        help='Comma-separated list of user IDs (e.g., "1,2,3")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for JSON results')
    
    args = parser.parse_args()
    
    tester = LoadTester(base_url=args.url)
    
    # Check health first
    print("Checking API health...")
    if not tester.test_health():
        print("❌ API health check failed. Is the server running?")
        return 1
    print("✅ API is healthy\n")
    
    # Parse user IDs if provided
    user_ids = None
    if args.user_ids:
        user_ids = [int(x.strip()) for x in args.user_ids.split(',')]
    
    results = []
    
    # Test places endpoint
    if args.endpoint in ['places', 'both']:
        places_result = tester.run_load_test(
            endpoint='places',
            num_requests=args.requests,
            concurrent=args.concurrent,
            user_ids=user_ids,
            top_k=10
        )
        tester.print_results(places_result)
        results.append(places_result)
    
    # Test people endpoint
    if args.endpoint in ['people', 'both']:
        people_result = tester.run_load_test(
            endpoint='people',
            num_requests=args.requests,
            concurrent=args.concurrent,
            user_ids=user_ids,
            top_k=10
        )
        tester.print_results(people_result)
        results.append(people_result)
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

