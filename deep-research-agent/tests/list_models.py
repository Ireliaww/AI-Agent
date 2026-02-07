#!/usr/bin/env python3
"""
List all available Gemini models and find embedding models
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Use the same import as your project
from google import genai
from google.genai import types

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not set")
    sys.exit(1)

print("🔍 Initializing Gemini client...")
client = genai.Client(api_key=api_key)

print("\n📋 Listing all available models...\n")

try:
    models = client.models.list()
    
    all_models = []
    embedding_models = []
    
    for model in models:
        model_name = model.name
        all_models.append(model_name)
        
        # Check for embedding models
        if 'embed' in model_name.lower() or 'embedding' in model_name.lower():
            embedding_models.append(model_name)
            print(f"✅ EMBEDDING MODEL: {model_name}")
            
            # Try to get supported methods
            if hasattr(model, 'supported_generation_methods'):
                print(f"   Methods: {model.supported_generation_methods}")
            if hasattr(model, 'description'):
                print(f"   Description: {model.description}")
            print()
    
    if not embedding_models:
        print("\n⚠️ No embedding models found in available models list.")
        print("\n📋 All available models:")
        for m in all_models[:20]:  # Show first 20
            print(f"  - {m}")
        if len(all_models) > 20:
            print(f"  ... and {len(all_models) - 20} more")
    else:
        print(f"\n✅ Found {len(embedding_models)} embedding model(s)")
        print("\n💡 Try using these in your code:")
        for m in embedding_models:
            print(f"  model_name=\"{m}\"")

except Exception as e:
    print(f"❌ Error listing models: {e}")
    print(f"\n🔍 Full error: {type(e).__name__}: {str(e)}")

print("\n\n🧪 Testing direct embedding call...")
test_models = [
    "models/text-embedding-004",
    "text-embedding-004",
    "models/embedding-001",
    "embedding-001",
]

for model_name in test_models:
    try:
        print(f"\nTrying: {model_name}...")
        result = client.models.embed_content(
            model=model_name,
            contents=["test text"]
        )
        print(f"  ✅ SUCCESS! Use: {model_name}")
        print(f"  Embedding dimension: {len(result.embeddings[0].values)}")
        break
    except Exception as e:
        error_msg = str(e)[:150]
        print(f"  ❌ Failed: {error_msg}")
