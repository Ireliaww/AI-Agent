#!/usr/bin/env python3
"""
Quick test script to verify bug fixes
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_title_extraction():
    """Test the improved _extract_paper_title method"""
    from agents.enhanced_research_agent import EnhancedResearchAgent
    from src.client import GeminiClient
    
    print("=" * 60)
    print("Testing Paper Title Extraction")
    print("=" * 60)
    
    gemini = GeminiClient()
    agent = EnhancedResearchAgent(gemini_client=gemini, use_mock=False)
    
    test_cases = [
        ("arxiv:1706.03762", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
        ('"Attention Is All You Need"', "Attention Is All You Need"),
        ("reproduce Attention Is All You Need paper", "Attention Is All You Need"),
        ("implement BERT from paper", "BERT"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected in test_cases:
        print(f"\n📝 Input: {input_text}")
        result = agent._extract_paper_title(input_text)
        
        # Normalize for comparison
        if expected in result or result in expected or expected.lower() in result.lower():
            print(f"✅ PASS - Got: {result}")
            passed += 1
        else:
            print(f"❌ FAIL - Expected: {expected}, Got: {result}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

async def test_agent_initialization():
    """Verify correct agents are being used"""
    print("\n")
    print("=" * 60)
    print("Testing Agent Initialization")
    print("=" * 60)
    
    try:
        from main import AIResearchAssistant
        from agents import EnhancedCodingAgent
        
        # Create assistant
        assistant = AIResearchAssistant(use_mock=False)
        await assistant.initialize()
        
        # Check coding agent type
        print(f"\n📊 Checking Coordinator's coding agent...")
        coding_agent = assistant.coordinator.coding_agent
        
        if isinstance(coding_agent, EnhancedCodingAgent):
            print(f"✅ PASS - Using EnhancedCodingAgent")
            print(f"   Has implement_from_paper: {hasattr(coding_agent, 'implement_from_paper')}")
            return True
        else:
            print(f"❌ FAIL - Using {type(coding_agent).__name__}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("\n🧪 AI-Agent Bug Fix Verification\n")
    
    # Test 1: Title extraction
    await test_title_extraction()
    
    # Test 2: Agent initialization
    result = await test_agent_initialization()
    
    print("\n" + "=" * 60)
    if result:
        print("🎉 All critical tests passed!")
        print("\nNext: Test full workflow with:")
        print('  python main.py -q "reproduce arxiv:1706.03762"')
    else:
        print("⚠️  Some tests failed - review output above")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
