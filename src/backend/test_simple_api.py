#!/usr/bin/env python3
"""
Test script for the Simple FastAPI Backend
Tests the /ask endpoint with sample student loan questions
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

# Sample student loan questions for testing
TEST_QUESTIONS = [
    "What are the requirements for federal student loan forgiveness?",
    "How do I apply for income-driven repayment plans?", 
    "What should I do if my loan servicer is not responding?",
    "What are the differences between federal and private student loans?",
    "How does loan consolidation work?",
    "What happens if I can't make my student loan payments?",
    "Are there any grants available for graduate students?",
    "How do I change my loan servicer?",
    "What is the Public Service Loan Forgiveness program?",
    "Can I get help with Nelnet payment issues?"
]

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint works: {data['message']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_ask_endpoint(question, max_length=None):
    """Test the /ask endpoint with a question"""
    print(f"\n🔍 Testing question: {question[:50]}...")
    
    payload = {"question": question}
    if max_length:
        payload["max_response_length"] = max_length
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/ask",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # 30 second timeout
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Question answered in {end_time - start_time:.2f}s")
            print(f"📝 Answer: {data['answer'][:200]}...")
            print(f"📊 Sources: {data['sources_count']}")
            print(f"✅ Success: {data['success']}")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

def test_api_info_endpoint():
    """Test the API info endpoint"""
    print("\n🔍 Testing API info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API info retrieved")
            print(f"📋 Knowledge base: {len(data['capabilities']['knowledge_base']['pdf_documents'])} PDFs")
            print(f"📊 Customer data: {data['capabilities']['knowledge_base']['customer_data']}")
            print(f"🎯 Retrieval method: {data['capabilities']['retrieval_method']}")
            return True
        else:
            print(f"❌ API info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API info error: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive API tests"""
    print("🚀 Starting comprehensive API test...")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health_endpoint():
        print("❌ Health check failed - API may not be running")
        return False
    
    # Test 2: Root endpoint
    if not test_root_endpoint():
        print("❌ Root endpoint failed")
        return False
    
    # Test 3: API info
    if not test_api_info_endpoint():
        print("❌ API info failed")
        return False
    
    # Test 4: Sample questions
    print(f"\n🔍 Testing {len(TEST_QUESTIONS)} sample questions...")
    successful_questions = 0
    
    for i, question in enumerate(TEST_QUESTIONS[:3], 1):  # Test first 3 questions
        print(f"\n--- Test Question {i}/{min(3, len(TEST_QUESTIONS))} ---")
        if test_ask_endpoint(question):
            successful_questions += 1
        time.sleep(1)  # Brief pause between requests
    
    # Test 5: Edge cases
    print(f"\n🔍 Testing edge cases...")
    
    # Empty question
    print("\n--- Testing empty question ---")
    payload = {"question": ""}
    try:
        response = requests.post(f"{BASE_URL}/ask", json=payload)
        if response.status_code == 400:
            print("✅ Empty question correctly rejected")
        else:
            print(f"❌ Empty question not handled properly: {response.status_code}")
    except Exception as e:
        print(f"❌ Empty question test error: {e}")
    
    # Very long question
    print("\n--- Testing very long question ---")
    long_question = "What are federal student loans? " * 100  # Very long question
    if test_ask_endpoint(long_question, max_length=100):
        print("✅ Long question handled successfully")
    
    # Results summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Successful questions: {successful_questions}/3")
    print(f"🎯 Success rate: {successful_questions/3*100:.1f}%")
    
    if successful_questions >= 2:
        print("🎉 API is working well!")
        return True
    else:
        print("⚠️ API has issues that need attention")
        return False

def quick_test():
    """Quick test with just one question"""
    print("🚀 Running quick API test...")
    
    if not test_health_endpoint():
        return False
    
    # Test one question
    test_question = "What should I do if I can't make my student loan payments?"
    return test_ask_endpoint(test_question)

if __name__ == "__main__":
    import sys
    
    print("Federal Student Loan API Test Script")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = quick_test()
    else:
        print("Run with --quick for a single question test")
        print("Running comprehensive test...\n")
        success = run_comprehensive_test()
    
    if success:
        print("\n🎉 Tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)