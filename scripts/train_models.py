"""
AirGuard AI — Model Training Script
Run this once to pre-train and save all ML models.

Usage:
    python scripts/train_models.py
"""

import sys
from pathlib import Path
import time

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.data_loader import load_city_day
from backend.ml_models import train_models


def main():
    print("=" * 60)
    print("  🌬️  AirGuard AI — Model Training")
    print("=" * 60)
    
    print("\n📊 Loading data...")
    start = time.time()
    df = load_city_day()
    print(f"   ✅ Loaded {len(df):,} records for {df['City'].nunique()} cities ({time.time()-start:.1f}s)")
    
    print("\n🤖 Training ML models...")
    print("   Training XGBoost, Random Forest, Gradient Boosting...")
    
    start = time.time()
    results, features, scaler = train_models(df)
    elapsed = time.time() - start
    
    print(f"\n✅ Training complete in {elapsed:.1f}s\n")
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 50)
    for model_name, res in results.items():
        print(f"{model_name:<20} {res['mae']:>8.2f} {res['rmse']:>8.2f} {res['r2']:>8.4f}")
    
    print(f"\n📁 Models saved to: {ROOT / 'models'}/")
    print("   - xgboost_model.pkl")
    print("   - rf_model.pkl")
    print("   - gb_model.pkl")
    print("   - scaler.pkl")
    print("   - features.pkl")
    
    print("\n🚀 Ready! You can now run:")
    print("   streamlit run app.py")
    print("   uvicorn backend.api:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
