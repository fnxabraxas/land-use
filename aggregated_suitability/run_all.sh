
echo === Computing suitability ===
python create_suitability.py

echo === Computing yield ratio ===
python create_yield-ratio.py

echo === Computing targets and sigma ===
python country_target.py

pyclean .
