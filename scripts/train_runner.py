import sys, os, time
sys.path.insert(0, '.')
from backend.data_loader import load_city_day
from backend.ml_models import train_models

print('Loading data...')
start = time.time()
df = load_city_day()
print('Loaded records:', len(df), 'Cities:', df['City'].nunique())
print('Time:', round(time.time()-start, 1), 's')

print('Training models...')
start = time.time()
results, features, scaler = train_models(df)
elapsed = time.time() - start
print('Training complete in', round(elapsed,1), 's')
for name, res in results.items():
    print(name, '| MAE:', round(res['mae'],1), '| RMSE:', round(res['rmse'],1), '| R2:', round(res['r2'],4))
print('Models saved to models/')
