# Market Vision AI for Stock Prediction

**Live Demo:** https://market-vision-ai.streamlit.app/
**Paper:** Hybrid Explainable AI (XAI) with CNN+LSTM+BiLSTM+Attention
**Code:** `app.py` Line 137-151 - 4 Algorithms Implemented

## 🧠 Algorithms (Q1 Proof)
- **Line 137:** CNN `nn.Conv1d(10,64,kernel_size=3,padding=1)` - 64 filters for chart patterns
- **Line 138:** LSTM `nn.LSTM(64,50)` - 50 units, 30 days memory
- **Line 139:** BiLSTM `nn.LSTM(..., bidirectional=True)` - Forward+Backward 100 units
- **Line 140:** Attention `nn.Linear(100,1)` + Softmax - XAI weighting

## Architecture
`HybridXAI: Input(10 features) -> CNN(64) -> LSTM(50) -> BiLSTM(50x2=100) -> Attention(100,1) -> FC(1)`

- Features: Close, Open, High, Low, Volume, SMA20, EMA20, RSI, MACD, VolSMA
- Seed: 42 (Deterministic)
- Optimizer: Adam lr=0.005, MSELoss, 3 epochs

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
