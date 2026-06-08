import pickle as pk
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pk.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pk.load(f)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Movie Review Sentiment Analyzer</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e2e8f0;font-family:'Space Grotesk',sans-serif;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem}
.container{max-width:640px;width:100%;text-align:center}
.badge{display:inline-flex;align-items:center;gap:6px;background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);color:#6366f1;padding:0.4rem 1rem;border-radius:999px;font-size:0.8rem;margin-bottom:1.5rem}
.dot{width:6px;height:6px;border-radius:50%;background:#10b981;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
h1{font-size:2.5rem;font-weight:700;margin-bottom:0.75rem;letter-spacing:-0.02em}
h1 span{background:linear-gradient(135deg,#6366f1,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.subtitle{color:#64748b;margin-bottom:2.5rem;line-height:1.6}
.card{background:#111118;border:1px solid #1e1e2e;border-radius:16px;padding:2rem;text-align:left}
label{display:block;font-size:0.85rem;color:#64748b;margin-bottom:0.5rem;font-weight:500}
textarea{width:100%;background:#0a0a0f;border:1px solid #1e1e2e;border-radius:10px;color:#e2e8f0;font-family:'Space Grotesk',sans-serif;font-size:0.95rem;padding:1rem;resize:vertical;min-height:120px;outline:none;transition:border-color 0.2s;line-height:1.6}
textarea:focus{border-color:#6366f1}
textarea::placeholder{color:#334155}
button{width:100%;background:#6366f1;color:#fff;border:none;border-radius:10px;padding:0.9rem;font-size:1rem;font-weight:600;font-family:'Space Grotesk',sans-serif;cursor:pointer;margin-top:1rem;transition:opacity 0.2s,transform 0.2s}
button:hover{opacity:0.9;transform:translateY(-1px)}
button:disabled{opacity:0.5;cursor:not-allowed;transform:none}
.result{margin-top:1.5rem;padding:1.25rem 1.5rem;border-radius:10px;display:none;align-items:center;gap:1rem;font-size:1.1rem;font-weight:600}
.result.positive{background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);color:#10b981}
.result.negative{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#ef4444}
.result-emoji{font-size:2rem}
.examples{margin-top:1.5rem}
.examples p{font-size:0.8rem;color:#475569;margin-bottom:0.5rem}
.example-btns{display:flex;flex-wrap:wrap;gap:0.5rem}
.example-btn{background:transparent;border:1px solid #1e1e2e;color:#64748b;padding:0.3rem 0.75rem;border-radius:6px;font-size:0.78rem;cursor:pointer;transition:border-color 0.2s,color 0.2s;font-family:'Space Grotesk',sans-serif}
.example-btn:hover{border-color:#6366f1;color:#6366f1}
.accuracy{display:flex;justify-content:center;gap:2rem;margin-top:2rem;padding-top:1.5rem;border-top:1px solid #1e1e2e;flex-wrap:wrap}
.stat{text-align:center}
.stat-num{font-size:1.5rem;font-weight:700;color:#e2e8f0}
.stat-label{font-size:0.75rem;color:#64748b;margin-top:2px}
.spinner{width:18px;height:18px;border:2px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.6s linear infinite;display:inline-block;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="container">
  <div class="badge"><span class="dot"></span> ML Model Live</div>
  <h1>Movie Review<br><span>Sentiment Analyzer</span></h1>
  <p class="subtitle">Enter any movie review and the AI model will instantly classify it as Positive or Negative using NLP and Naive Bayes classification.</p>
  <div class="card">
    <label for="review">Movie Review</label>
    <textarea id="review" placeholder="e.g. This movie was absolutely amazing! The acting was brilliant..."></textarea>
    <button onclick="predict()" id="btn">Analyze Sentiment</button>
    <div class="result" id="result">
      <span class="result-emoji" id="emoji"></span>
      <div>
        <div id="sentiment-text"></div>
        <div style="font-size:0.8rem;opacity:0.7;font-weight:400;margin-top:2px">Sentiment detected</div>
      </div>
    </div>
    <div class="examples">
      <p>Try an example:</p>
      <div class="example-btns">
        <button class="example-btn" onclick="setExample('An absolute masterpiece. One of the best films I have ever seen.')">Positive example</button>
        <button class="example-btn" onclick="setExample('Terrible waste of time. The plot made no sense and the acting was awful.')">Negative example</button>
        <button class="example-btn" onclick="setExample('The cinematography was stunning but the story felt hollow and predictable.')">Mixed example</button>
      </div>
    </div>
  </div>
  <div class="accuracy">
    <div class="stat"><div class="stat-num">85%</div><div class="stat-label">Accuracy</div></div>
    <div class="stat"><div class="stat-num">0.87</div><div class="stat-label">F1 Score</div></div>
    <div class="stat"><div class="stat-num">Naive Bayes</div><div class="stat-label">Algorithm</div></div>
    <div class="stat"><div class="stat-num">TF-IDF</div><div class="stat-label">Vectorizer</div></div>
  </div>
</div>
<script>
function setExample(text){document.getElementById('review').value=text}
async function predict(){
  const review=document.getElementById('review').value.trim();
  if(!review){alert('Please enter a movie review');return}
  const btn=document.getElementById('btn');
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Analyzing...';
  const result=document.getElementById('result');
  result.style.display='none';
  try{
    const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({review})});
    const data=await res.json();
    if(data.error){alert('Server error: '+data.error);return;}
    result.className='result '+(data.sentiment==='Positive'?'positive':'negative');
    result.style.display='flex';
    document.getElementById('emoji').textContent=data.emoji;
    document.getElementById('sentiment-text').textContent=data.sentiment+' Review';
  }catch(e){alert('Error: '+e.message)}
  finally{btn.disabled=false;btn.innerHTML='Analyze Sentiment'}
}
document.getElementById('review').addEventListener('keydown',function(e){if(e.ctrlKey&&e.key==='Enter')predict()});
</script>
</body>
</html>"""

@app.route('/')
def index():
    return HTML

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data.get('review', '')
        if not review.strip():
            return jsonify({'error': 'Please enter a review'}), 400
        review_vectorized = scaler.transform([review])
        result = model.predict(review_vectorized)
        sentiment = 'Positive' if result[0] == 1 else 'Negative'
        emoji = '😊' if result[0] == 1 else '😞'
        return jsonify({'sentiment': sentiment, 'emoji': emoji})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)