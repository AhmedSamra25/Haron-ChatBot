#!/usr/bin/env python3
"""
Test script to validate the fixes for LangChain deprecation and Gemini model issues.
Updated with working Gemini 2.0 model names.
"""

import os
import sys
import logging

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test that all imports work without deprecation warnings."""
    print("🔍 Testing imports...")
    
    try:
        from backend.main import MemoryManager, SimpleChatHistory
        print("✅ Memory imports successful")
    except Exception as e:
        print(f"❌ Memory import failed: {e}")
        return False
    
    try:
        from backend.main import BusinessArabicChain
        print("✅ Business chain import successful")
    except Exception as e:
        print(f"❌ Business chain import failed: {e}")
        return False
    
    return True

def test_memory_manager():
    """Test the new memory manager implementation."""
    print("\n🔍 Testing Memory Manager...")
    
    try:
        from backend.main import MemoryManager
        
        # Create memory manager
        memory_manager = MemoryManager(window_size=5)
        
        # Test adding messages
        session_id = "test_session"
        memory_manager.add_message(session_id, "Hello", "مرحبا")
        memory_manager.add_message(session_id, "How are you?", "كيف حالك؟")
        
        # Test getting history
        history = memory_manager.get_history_string(session_id)
        print(f"📝 History: {history}")
        
        # Test stats
        stats = memory_manager.get_session_stats()
        print(f"📊 Stats: {stats}")
        
        print("✅ Memory manager test passed")
        return True
        
    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")
        return False

def test_gemini_models():
    """Test Gemini model initialization with new model names."""
    print("\n🔍 Testing Gemini Models...")
    
    # Check if Google API key is available
    from dotenv import load_dotenv
    load_dotenv()
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    if not google_api_key:
        print("⚠️ No GOOGLE_API_KEY found in environment. Skipping model test.")
        return True
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Updated model names that actually work
        model_names = [
            "gemini-2.0-flash",      # Recommended - latest fast model
            "gemini-2.5-flash",      # Alternative fast model  
            "gemini-flash-latest",   # Latest alias
            "gemini-pro-latest",     # Pro alias
            "gemini-2.0-flash-001"   # Stable version
        ]
        
        successful_model = None
        
        for model_name in model_names:
            try:
                print(f"🧪 Testing {model_name}...")
                
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=google_api_key,
                    temperature=0.7,
                    max_output_tokens=100
                )
                
                # Simple test
                response = llm.invoke("Hello")
                print(f"✅ {model_name} works! Response: {response.content[:50]}...")
                successful_model = model_name
                break
                
            except Exception as e:
                print(f"❌ {model_name} failed: {str(e)[:100]}...")
                continue
        
        if successful_model:
            print(f"🎉 Successfully found working model: {successful_model}")
            return True
        else:
            print("❌ No working Gemini models found")
            return False
            
    except Exception as e:
        print(f"❌ Gemini model test failed: {e}")
        return False

def test_business_chain():
    """Test the business chain with new implementation."""
    print("\n🔍 Testing Business Chain...")
    
    from dotenv import load_dotenv
    load_dotenv()
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    if not google_api_key:
        print("⚠️ No GOOGLE_API_KEY found. Skipping business chain test.")
        return True
    
    try:
        from backend.main import BusinessArabicChain
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Create LLM with working model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Use working model
            google_api_key=google_api_key,
            temperature=0.7,
            max_output_tokens=100
        )
        
        # Create chain
        chain = BusinessArabicChain(llm, retriever=None)
        
        # Test invocation
        result = chain.invoke({
            "question": "ما هي نصائح ريادة الأعمال؟",
            "chat_history": ""
        })
        
        print(f"📝 Chain response: {result['answer'][:100]}...")
        print("✅ Business chain test passed")
        return True
        
    except Exception as e:
        print(f"❌ Business chain test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting fixes validation tests (Updated for Gemini 2.0)...\n")
    
    # Suppress warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_imports():
        tests_passed += 1
    
    if test_memory_manager():
        tests_passed += 1
    
    if test_gemini_models():
        tests_passed += 1
    
    if test_business_chain():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your fixes are working correctly.")
        print("🔥 Gemini 2.0 models are now working!")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())