#!/usr/bin/env python3
"""
Comprehensive test suite for the production-ready financial forecasting system.
"""

import requests
import json
import time
import pandas as pd
from typing import Dict, List, Any

class ProductionSystemTester:
    """Test suite for the production financial forecasting system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print("🧪 RUNNING COMPREHENSIVE PRODUCTION SYSTEM TESTS")
        print("=" * 80)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Model Info", self.test_model_info),
            ("Root Endpoint", self.test_root_endpoint),
            ("Guided Forecast - Basic", self.test_guided_forecast_basic),
            ("Guided Forecast - Advanced", self.test_guided_forecast_advanced),
            ("Guided Forecast - Validation", self.test_input_validation),
            ("CSV Upload", self.test_csv_upload),
            ("Chat Interface", self.test_chat_interface),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n🔍 Testing: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
                print(f"   {status}: {result.get('message', 'No message')}")
            except Exception as e:
                results[test_name] = {
                    "success": False,
                    "error": str(e),
                    "message": f"Test failed with exception: {str(e)}"
                }
                print(f"   ❌ FAIL: {str(e)}")
        
        # Summary
        passed = sum(1 for r in results.values() if r.get("success", False))
        total = len(results)
        
        print(f"\n{'='*80}")
        print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
        print(f"{'='*80}")
        
        return {
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": f"{(passed/total)*100:.1f}%"
            },
            "results": results
        }
    
    def test_health_check(self) -> Dict[str, Any]:
        """Test the health check endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": f"Health check passed. Status: {data.get('overall_health', 'Unknown')}",
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "message": f"Health check failed with status {response.status_code}",
                    "data": response.text
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Health check failed: {str(e)}"
            }
    
    def test_model_info(self) -> Dict[str, Any]:
        """Test the model info endpoint."""
        try:
            response = requests.get(f"{self.base_url}/model-info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": f"Model info retrieved. R²: {data.get('overall_r2', 'Unknown')}",
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "message": f"Model info failed with status {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Model info failed: {str(e)}"
            }
    
    def test_root_endpoint(self) -> Dict[str, Any]:
        """Test the root endpoint."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": f"Root endpoint working. Version: {data.get('version', 'Unknown')}",
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "message": f"Root endpoint failed with status {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Root endpoint failed: {str(e)}"
            }
    
    def test_guided_forecast_basic(self) -> Dict[str, Any]:
        """Test basic guided forecast."""
        try:
            payload = {
                "company_name": "Test Company",
                "current_arr": 5000000,
                "net_new_arr": 1000000
            }
            
            response = requests.post(
                f"{self.base_url}/guided_forecast",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": f"Basic forecast successful. Model: {data.get('model_used', 'Unknown')}",
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "message": f"Basic forecast failed with status {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Basic forecast failed: {str(e)}"
            }
    
    def test_guided_forecast_advanced(self) -> Dict[str, Any]:
        """Test advanced guided forecast with enhanced mode."""
        try:
            payload = {
                "company_name": "Advanced Test Company",
                "current_arr": 10000000,
                "net_new_arr": 2000000,
                "enhanced_mode": True,
                "sector": "Data & Analytics",
                "country": "United States",
                "currency": "USD",
                "advanced_mode": True,
                "advanced_metrics": {
                    "gross_margin": 85.0,
                    "headcount": 100
                }
            }
            
            response = requests.post(
                f"{self.base_url}/guided_forecast",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": f"Advanced forecast successful. Model: {data.get('model_used', 'Unknown')}",
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "message": f"Advanced forecast failed with status {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Advanced forecast failed: {str(e)}"
            }
    
    def test_input_validation(self) -> Dict[str, Any]:
        """Test input validation."""
        test_cases = [
            {
                "name": "Negative ARR",
                "payload": {"current_arr": -1000, "net_new_arr": 100},
                "should_fail": True
            },
            {
                "name": "Negative Net New ARR",
                "payload": {"current_arr": 1000, "net_new_arr": -100},
                "should_fail": True
            },
            {
                "name": "Net New ARR > Current ARR",
                "payload": {"current_arr": 1000, "net_new_arr": 2000},
                "should_fail": True
            },
            {
                "name": "Extreme Growth Rate",
                "payload": {"current_arr": 1000, "net_new_arr": 10000},
                "should_fail": True
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/guided_forecast",
                    json=test_case["payload"],
                    timeout=10
                )
                
                if test_case["should_fail"]:
                    success = response.status_code == 400
                    message = "Validation correctly rejected invalid input" if success else "Validation failed to reject invalid input"
                else:
                    success = response.status_code == 200
                    message = "Validation correctly accepted valid input" if success else "Validation incorrectly rejected valid input"
                
                results.append({
                    "test": test_case["name"],
                    "success": success,
                    "message": message
                })
                
            except Exception as e:
                results.append({
                    "test": test_case["name"],
                    "success": False,
                    "message": f"Test failed: {str(e)}"
                })
        
        passed = sum(1 for r in results if r["success"])
        return {
            "success": passed == len(results),
            "message": f"Input validation: {passed}/{len(results)} tests passed",
            "data": results
        }
    
    def test_csv_upload(self) -> Dict[str, Any]:
        """Test CSV upload functionality."""
        try:
            # Create a test CSV
            test_data = pd.DataFrame({
                'Quarter': ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023'],
                'ARR_End_of_Quarter': [1000000, 1200000, 1400000, 1600000],
                'Quarterly_Net_New_ARR': [0, 200000, 200000, 200000],
                'Headcount': [10, 12, 14, 16],
                'Gross_Margin_Percent': [80, 81, 82, 83]
            })
            
            # Save to temporary file
            test_file = "test_upload.csv"
            test_data.to_csv(test_file, index=False)
            
            # Upload the file
            with open(test_file, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.base_url}/predict_csv",
                    files=files,
                    data={'model': 'lightgbm'},
                    timeout=30
                )
            
            # Clean up
            import os
            os.remove(test_file)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": f"CSV upload successful. Model: {data.get('model_used', 'Unknown')}",
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "message": f"CSV upload failed with status {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"CSV upload failed: {str(e)}"
            }
    
    def test_chat_interface(self) -> Dict[str, Any]:
        """Test the chat interface."""
        try:
            payload = {
                "message": "My ARR is $5M and net new ARR is $1M. Can you forecast my growth?",
                "name": "Test User"
            }
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": "Chat interface working",
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "message": f"Chat interface failed with status {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Chat interface failed: {str(e)}"
            }
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid requests."""
        try:
            # Test with invalid JSON
            response = requests.post(
                f"{self.base_url}/guided_forecast",
                data="invalid json",
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            # Should return 422 (validation error) or 400 (bad request)
            success = response.status_code in [400, 422]
            
            return {
                "success": success,
                "message": f"Error handling {'working' if success else 'failed'}. Status: {response.status_code}",
                "data": {"status_code": response.status_code, "response": response.text}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error handling test failed: {str(e)}"
            }
    
    def test_performance(self) -> Dict[str, Any]:
        """Test system performance with multiple requests."""
        try:
            payload = {
                "company_name": "Performance Test",
                "current_arr": 5000000,
                "net_new_arr": 1000000
            }
            
            # Test 5 concurrent requests
            start_time = time.time()
            response_times = []
            
            for i in range(5):
                request_start = time.time()
                response = requests.post(
                    f"{self.base_url}/guided_forecast",
                    json=payload,
                    timeout=30
                )
                request_time = time.time() - request_start
                response_times.append(request_time)
                
                if response.status_code != 200:
                    return {
                        "success": False,
                        "message": f"Performance test failed on request {i+1}"
                    }
            
            total_time = time.time() - start_time
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            return {
                "success": True,
                "message": f"Performance test passed. Avg: {avg_response_time:.2f}s, Max: {max_response_time:.2f}s",
                "data": {
                    "total_time": total_time,
                    "avg_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "response_times": response_times
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Performance test failed: {str(e)}"
            }

def main():
    """Run the comprehensive test suite."""
    tester = ProductionSystemTester()
    results = tester.run_all_tests()
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Test results saved to test_results.json")
    
    # Final assessment
    success_rate = float(results["summary"]["success_rate"].replace("%", ""))
    if success_rate >= 90:
        print(f"\n🎉 EXCELLENT! System is production-ready! ({success_rate}% tests passed)")
    elif success_rate >= 80:
        print(f"\n✅ GOOD! System is mostly ready with minor issues ({success_rate}% tests passed)")
    elif success_rate >= 70:
        print(f"\n⚠️  FAIR! System needs some fixes before deployment ({success_rate}% tests passed)")
    else:
        print(f"\n❌ POOR! System needs significant work before deployment ({success_rate}% tests passed)")

if __name__ == "__main__":
    main()


