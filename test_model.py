"""
Quick test to verify HuggingFace model loading
"""
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 60)
print("Testing FinBERT Model Loading from HuggingFace")
print("=" * 60)

try:
    from finbert import FinBERTEngine
    
    model_name = "Arstacity/finbert-finetuned"
    print(f"\nLoading model: {model_name}")
    print("This will download the model (~400MB) on first run...")
    print("Please wait...\n")
    
    finbert = FinBERTEngine(model_name=model_name)
    
    print("✓ Model loaded successfully!\n")
    
    # Test prediction
    print("Testing sentiment prediction:")
    test_texts = [
        "Stock market surged 5% today on positive earnings",
        "Company faces bankruptcy amid financial troubles",
        "Market remained stable with no significant changes"
    ]
    
    results = finbert.predict(test_texts)
    
    print("\nResults:")
    print("-" * 60)
    for text, result in zip(test_texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (confidence: {result['confidence']:.2%})")
        print(f"Numeric: {result['numeric']}")
        print()
    
    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print(f"\nYou can now run the full pipeline:")
    print(f"  python tests/test.py --stages 1A 1B 1C")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you have internet connection")
    print("2. Verify transformers is installed: pip install transformers")
    print("3. Check if huggingface_hub is installed: pip install huggingface_hub")
    import traceback
    traceback.print_exc()
