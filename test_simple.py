#!/usr/bin/env python3
"""
Simple test script to verify humanizer functionality
"""

import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_humanizers():
    """Test both humanizer models."""
    try:
        from humanizers import BalancedHumanizer, AggressiveHumanizer
        print("✅ Successfully imported humanizers")
    except ImportError as e:
        print(f"❌ Failed to import humanizers: {e}")
        return False

    # Test text
    test_text = "Furthermore, it is important to note that artificial intelligence is transforming industries. Moreover, the comprehensive analysis demonstrates significant improvements."
    
    print(f"\nOriginal text: {test_text}")
    
    # Test Balanced Humanizer
    print("\n🔄 Testing Balanced Humanizer...")
    try:
        balanced = BalancedHumanizer(load_datasets=False)
        humanized_balanced, stats_balanced = balanced.humanize(test_text)
        print(f"✅ Balanced result: {humanized_balanced}")
        print(f"📊 Improvement: {stats_balanced['improvement']:.1f}%")
    except Exception as e:
        print(f"❌ Balanced humanizer failed: {e}")
        return False
    
    # Test Aggressive Humanizer
    print("\n🔄 Testing Aggressive Humanizer...")
    try:
        aggressive = AggressiveHumanizer(load_datasets=False)
        humanized_aggressive, stats_aggressive = aggressive.humanize(test_text)
        print(f"✅ Aggressive result: {humanized_aggressive}")
        print(f"📊 Improvement: {stats_aggressive['improvement']:.1f}%")
    except Exception as e:
        print(f"❌ Aggressive humanizer failed: {e}")
        return False
    
    return True

def test_cli_import():
    """Test CLI script import."""
    try:
        # Test if CLI script can be imported
        cli_path = Path(__file__).parent / "cli" / "humanize_cli.py"
        if cli_path.exists():
            print("✅ CLI script exists")
        else:
            print("❌ CLI script not found")
            return False
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False
    
    return True

def test_api_import():
    """Test API script import."""
    try:
        # Test if API script can be imported
        api_path = Path(__file__).parent / "api" / "app.py"
        if api_path.exists():
            print("✅ API script exists")
        else:
            print("❌ API script not found")
            return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧠 AI Text Humanizer Simple Test")
    print("=" * 40)
    
    # Test humanizers
    humanizer_success = test_humanizers()
    
    # Test CLI
    cli_success = test_cli_import()
    
    # Test API
    api_success = test_api_import()
    
    print("\n" + "=" * 40)
    print("📊 TEST RESULTS:")
    print(f"Humanizers: {'✅ PASSED' if humanizer_success else '❌ FAILED'}")
    print(f"CLI Script: {'✅ PASSED' if cli_success else '❌ FAILED'}")
    print(f"API Script: {'✅ PASSED' if api_success else '❌ FAILED'}")
    
    if humanizer_success and cli_success and api_success:
        print("\n🎉 All basic tests passed!")
        print("\nTo test the CLI manually, run:")
        print('python cli/humanize_cli.py --model balanced --input "Your text here"')
        print("\nTo test the API manually, run:")
        print('python api/app.py')
        print('Then visit http://localhost:5000 for documentation')
    else:
        print("\n❌ Some tests failed.")
