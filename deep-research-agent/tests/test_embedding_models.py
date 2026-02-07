#!/usr/bin/env python3
"""
Test script to list all available Gemini models
"""
import os
from google import genai

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not set")
    exit(1)

client = genai.Client(api_key=api_key)

print("🔍 Listing all available Gemini models...\n")

try:
    # List all models
    models = client.models.list()
    
    embedding_models = []
    
    for model in models:
        # Check if it's an embedding model
        if 'embed' in model.name.lower():
            embedding_models.append(model.name)
            print(f"✅ Embedding Model: {model.name}")
            if hasattr(model, 'supported_generation_methods'):
                print(f"   Supported methods: {model.supported_generation_methods}")
            print()
    
    if not embedding_models:
        print("\n⚠️ No embedding models found. All models:")
        for model in models:
            print(f"  - {model.name}")
    else:
        print(f"\n📊 Total embedding models found: {len(embedding_models)}")
        print("\n💡 Try using one of these names (without 'models/' prefix):")
        for model_name in embedding_models:
            clean_name = model_name.replace('models/', '')
            print(f"  - {clean_name}")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\n🔍 Trying alternative method...")
    
    # Try a direct test
    test_models = [
        "text-embedding-004",
        "embedding-001", 
        "models/text-embedding-004",
        "models/embedding-001",
    ]
    
    for model in test_models:
        try:
            result = client.models.embed_content(
                model=model,
                contents=["test"]
            )
            print(f"✅ SUCCESS: {model}")
            break
        except Exception as e:
            print(f"❌ FAILED: {model} - {str(e)[:100]}")
