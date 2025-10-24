#!/usr/bin/env python3
"""
Integration test script for AI Text Humanizer CLI and API
Tests both the CLI interface and Flask API functionality.
"""

import subprocess
import requests
import time
import json
import os
import sys
from pathlib import Path

def test_cli():
    """Test CLI functionality."""
    print("🧪 Testing CLI functionality...")
    
    test_text = "Furthermore, it is important to note that artificial intelligence is transforming industries. Moreover, the comprehensive analysis demonstrates significant improvements in various sectors."
    
    # Test balanced model
    print("  Testing balanced model...")
    try:
        result = subprocess.run([
            sys.executable, "cli/humanize_cli.py",
            "--model", "balanced",
            "--input", test_text,
            "--stats",
            "--quiet"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("  ✅ Balanced model CLI test passed")
        else:
            print(f"  ❌ Balanced model CLI test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ CLI test error: {e}")
        return False
    
    # Test aggressive model
    print("  Testing aggressive model...")
    try:
        result = subprocess.run([
            sys.executable, "cli/humanize_cli.py",
            "--model", "aggressive",
            "--input", test_text,
            "--quiet"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("  ✅ Aggressive model CLI test passed")
        else:
            print(f"  ❌ Aggressive model CLI test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ CLI test error: {e}")
        return False
    
    # Test file input/output
    print("  Testing file input/output...")
    try:
        # Create test input file
        test_file = Path(__file__).parent / "test_input.txt"
        output_file = Path(__file__).parent / "test_output.txt"
        
        with open(test_file, 'w') as f:
            f.write(test_text)
        
        result = subprocess.run([
            sys.executable, "cli/humanize_cli.py",
            "--model", "balanced",
            "--file", str(test_file),
            "--output", str(output_file),
            "--quiet"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0 and output_file.exists():
            print("  ✅ File I/O CLI test passed")
            # Clean up
            test_file.unlink()
            output_file.unlink()
        else:
            print(f"  ❌ File I/O CLI test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ File I/O test error: {e}")
        return False
    
    return True

def start_api_server():
    """Start the API server in background."""
    print("🚀 Starting API server...")
    try:
        process = subprocess.Popen([
            sys.executable, "api/app.py"
        ], cwd=Path(__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(5)
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code == 200:
                print("  ✅ API server started successfully")
                return process
            else:
                print(f"  ❌ API server health check failed: {response.status_code}")
                process.terminate()
                return None
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Could not connect to API server: {e}")
            process.terminate()
            return None
    except Exception as e:
        print(f"  ❌ Error starting API server: {e}")
        return None

def test_api():
    """Test API functionality."""
    print("🧪 Testing API functionality...")
    
    base_url = "http://localhost:5000"
    test_text = "Furthermore, it is important to note that artificial intelligence is transforming industries."
    
    # Test health endpoint
    print("  Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("  ✅ Health endpoint test passed")
        else:
            print(f"  ❌ Health endpoint test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Health endpoint error: {e}")
        return False
    
    # Test documentation endpoint
    print("  Testing documentation endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("  ✅ Documentation endpoint test passed")
        else:
            print(f"  ❌ Documentation endpoint test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Documentation endpoint error: {e}")
        return False
    
    # Test balanced model
    print("  Testing balanced model API...")
    try:
        response = requests.post(f"{base_url}/humanize", json={
            "model": "balanced",
            "text": test_text,
            "include_stats": True
        })
        
        if response.status_code == 200:
            data = response.json()
            if "humanized_text" in data and "stats" in data:
                print("  ✅ Balanced model API test passed")
            else:
                print(f"  ❌ Balanced model API response missing fields: {data}")
                return False
        else:
            print(f"  ❌ Balanced model API test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"  ❌ Balanced model API error: {e}")
        return False
    
    # Test aggressive model
    print("  Testing aggressive model API...")
    try:
        response = requests.post(f"{base_url}/humanize", json={
            "model": "aggressive",
            "text": test_text
        })
        
        if response.status_code == 200:
            data = response.json()
            if "humanized_text" in data:
                print("  ✅ Aggressive model API test passed")
            else:
                print(f"  ❌ Aggressive model API response missing fields: {data}")
                return False
        else:
            print(f"  ❌ Aggressive model API test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"  ❌ Aggressive model API error: {e}")
        return False
    
    # Test error handling
    print("  Testing API error handling...")
    try:
        # Test invalid model
        response = requests.post(f"{base_url}/humanize", json={
            "model": "invalid",
            "text": test_text
        })
        
        if response.status_code == 400:
            print("  ✅ Invalid model error handling test passed")
        else:
            print(f"  ❌ Invalid model error handling failed: {response.status_code}")
            return False
        
        # Test missing text
        response = requests.post(f"{base_url}/humanize", json={
            "model": "balanced"
        })
        
        if response.status_code == 400:
            print("  ✅ Missing text error handling test passed")
        else:
            print(f"  ❌ Missing text error handling failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error handling test error: {e}")
        return False
    
    return True

def main():
    """Run integration tests."""
    print("🧠 AI Text Humanizer Integration Tests")
    print("=" * 50)
    
    # Test CLI
    cli_success = test_cli()
    
    print()
    
    # Test API
    api_process = start_api_server()
    api_success = False
    
    if api_process:
        try:
            api_success = test_api()
        finally:
            print("🛑 Stopping API server...")
            api_process.terminate()
            api_process.wait()
    
    print()
    print("=" * 50)
    print("📊 TEST RESULTS:")
    print(f"CLI Tests: {'✅ PASSED' if cli_success else '❌ FAILED'}")
    print(f"API Tests: {'✅ PASSED' if api_success else '❌ FAILED'}")
    
    if cli_success and api_success:
        print("🎉 All tests passed! CLI and API are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
