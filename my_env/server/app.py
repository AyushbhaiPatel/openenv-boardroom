# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the OpenBoardroom Environment.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from fastapi.responses import HTMLResponse

try:
    from my_env.models import BoardroomAction, BoardroomObservation
    from my_env.server.boardroom_environment import BoardroomEnvironment
except ImportError:  # pragma: no cover — inside Docker PYTHONPATH=/app/env
    from models import BoardroomAction, BoardroomObservation
    from boardroom_environment import BoardroomEnvironment


app = create_app(
    BoardroomEnvironment,
    BoardroomAction,
    BoardroomObservation,
    env_name="boardroom",
    max_concurrent_envs=1,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OpenBoardroom — Try the AI Environment</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;display:flex;flex-direction:column}

/* Header */
.header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px;display:flex;align-items:center;gap:12px}
.header-title{font-size:1.2rem;font-weight:700;color:#58a6ff}
.header-sub{font-size:0.82rem;color:#8b949e;margin-top:2px}
.live-dot{width:8px;height:8px;background:#3fb950;border-radius:50%;display:inline-block;margin-right:4px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* Steps wizard */
.wizard{max-width:680px;margin:32px auto;padding:0 20px;flex:1}
.step-indicator{display:flex;align-items:center;margin-bottom:28px}
.step-dot{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;flex-shrink:0;transition:all .3s}
.step-dot.done{background:#3fb950;color:#000}
.step-dot.active{background:#58a6ff;color:#000}
.step-dot.pending{background:#21262d;color:#8b949e;border:1px solid #30363d}
.step-line{flex:1;height:2px;background:#21262d;margin:0 8px}
.step-line.done{background:#3fb950}

/* Cards */
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px;margin-bottom:16px}
.card-title{font-size:1rem;font-weight:600;color:#e6edf3;margin-bottom:4px}
.card-sub{font-size:0.82rem;color:#8b949e;margin-bottom:18px}

/* Difficulty picker */
.diff-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:18px}
.diff-card{background:#0d1117;border:2px solid #30363d;border-radius:10px;padding:14px 10px;text-align:center;cursor:pointer;transition:all .2s}
.diff-card:hover{border-color:#58a6ff;transform:translateY(-2px)}
.diff-card.selected{border-color:#58a6ff;background:#0c2135}
.diff-card .diff-icon{font-size:1.6rem;margin-bottom:6px}
.diff-card .diff-name{font-weight:700;font-size:0.9rem}
.diff-card .diff-desc{font-size:0.72rem;color:#8b949e;margin-top:4px;line-height:1.4}
.diff-card .diff-steps{font-size:0.7rem;color:#58a6ff;margin-top:6px;font-weight:600}

/* Big button */
.btn{width:100%;padding:13px;border:none;border-radius:8px;font-size:0.95rem;font-weight:600;cursor:pointer;transition:all .2s}
.btn-primary{background:#238636;color:#fff}
.btn-primary:hover{background:#2ea043}
.btn-primary:disabled{background:#21262d;color:#484f58;cursor:not-allowed}
.btn-secondary{background:#21262d;color:#c9d1d9;border:1px solid #30363d;margin-top:8px}
.btn-secondary:hover{background:#30363d}

/* Action picker */
.action-grid{display:grid;gap:8px;margin-bottom:16px}
.action-btn{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px 16px;cursor:pointer;text-align:left;transition:all .2s;display:flex;align-items:center;gap:12px}
.action-btn:hover{border-color:#58a6ff;background:#0c2135}
.action-btn.selected{border-color:#58a6ff;background:#0c2135}
.action-icon{font-size:1.2rem;flex-shrink:0}
.action-name{font-weight:600;font-size:0.88rem;color:#e6edf3}
.action-hint{font-size:0.75rem;color:#8b949e;margin-top:2px}

/* Param field */
.param-field{margin-bottom:12px}
.param-field label{display:block;font-size:0.78rem;color:#8b949e;margin-bottom:5px;font-weight:500}
.param-field select,.param-field input{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;font-size:0.88rem;padding:9px 11px;outline:none;transition:border-color .2s}
.param-field select:focus,.param-field input:focus{border-color:#58a6ff}
.param-field textarea{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;font-size:0.82rem;padding:9px 11px;outline:none;resize:vertical;min-height:70px;font-family:monospace;transition:border-color .2s}
.param-field textarea:focus{border-color:#58a6ff}

/* Result */
.result-box{background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:16px;margin-top:4px}
.result-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #21262d;font-size:0.85rem}
.result-row:last-child{border-bottom:none}
.result-label{color:#8b949e}
.result-val{font-weight:600;color:#e6edf3}
.result-val.green{color:#3fb950}
.result-val.red{color:#f85149}
.result-val.blue{color:#58a6ff}

.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.metric-tile{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 12px}
.metric-tile .mname{font-size:0.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em}
.metric-tile .mval{font-size:1rem;font-weight:700;color:#58a6ff;margin-top:3px}

.feedback-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-top:10px;font-size:0.85rem;color:#c9d1d9;line-height:1.5}
.feedback-label{font-size:0.7rem;color:#8b949e;font-weight:600;margin-bottom:4px;text-transform:uppercase}

.score-card{background:#0c2135;border:1px solid #1f6feb;border-radius:10px;padding:20px;text-align:center;margin-top:10px}
.score-num{font-size:2.8rem;font-weight:800;color:#58a6ff}
.score-label{font-size:0.82rem;color:#8b949e;margin-top:4px}

.progress-bar-wrap{background:#21262d;border-radius:20px;height:6px;margin:10px 0 4px}
.progress-bar{background:linear-gradient(90deg,#238636,#3fb950);height:6px;border-radius:20px;transition:width .4s}

.step-counter{font-size:0.78rem;color:#8b949e;text-align:right;margin-bottom:8px}

.toast{position:fixed;bottom:24px;right:24px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 18px;font-size:0.85rem;display:none;z-index:99;max-width:300px}
.toast.show{display:block}
.toast.success{border-color:#238636;color:#3fb950}
.toast.error{border-color:#f85149;color:#f85149}

.hidden{display:none}
.loading{opacity:.6;pointer-events:none}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="header-title">📊 OpenBoardroom &nbsp;<span style="font-size:0.75rem;background:#1a3a1a;color:#3fb950;padding:2px 8px;border-radius:20px;font-weight:500"><span class="live-dot"></span>Live</span></div>
    <div class="header-sub">You are the Chief Data Officer of a SaaS company. Make smart decisions using data.</div>
  </div>
</div>

<div class="wizard" id="wizard">

  <!-- Step indicator -->
  <div class="step-indicator">
    <div class="step-dot active" id="dot-1">1</div>
    <div class="step-line" id="line-1"></div>
    <div class="step-dot pending" id="dot-2">2</div>
    <div class="step-line" id="line-2"></div>
    <div class="step-dot pending" id="dot-3">3</div>
  </div>

  <!-- ── SCREEN 1: Choose difficulty ── -->
  <div id="screen-1">
    <div class="card">
      <div class="card-title">Choose your challenge</div>
      <div class="card-sub">Pick a difficulty. The AI agent (or you!) will try to solve the business problem.</div>
      <div class="diff-grid">
        <div class="diff-card selected" id="diff-easy" onclick="pickDiff('easy')">
          <div class="diff-icon">🟢</div>
          <div class="diff-name">Easy</div>
          <div class="diff-desc">Find the Growth Bottleneck</div>
          <div class="diff-steps">10 steps · Clean data</div>
        </div>
        <div class="diff-card" id="diff-medium" onclick="pickDiff('medium')">
          <div class="diff-icon">🟡</div>
          <div class="diff-name">Medium</div>
          <div class="diff-desc">Diagnose the Revenue Drop</div>
          <div class="diff-steps">20 steps · Noisy data</div>
        </div>
        <div class="diff-card" id="diff-hard" onclick="pickDiff('hard')">
          <div class="diff-icon">🔴</div>
          <div class="diff-name">Hard</div>
          <div class="diff-desc">Should We Launch Feature X?</div>
          <div class="diff-steps">30 steps · Misleading signals</div>
        </div>
      </div>
      <button class="btn btn-primary" onclick="startGame()">▶ Start Episode</button>
    </div>
  </div>

  <!-- ── SCREEN 2: Play ── -->
  <div id="screen-2" class="hidden">
    <!-- Objective banner -->
    <div id="objective-banner" style="background:#0c2135;border:1px solid #1f6feb;border-radius:10px;padding:12px 16px;margin-bottom:14px;font-size:0.85rem;color:#93c5fd">
      🎯 <span id="objective-text">Loading...</span>
    </div>

    <!-- Progress -->
    <div class="step-counter">Step <span id="cur-step">0</span> / <span id="max-step">10</span></div>
    <div class="progress-bar-wrap"><div class="progress-bar" id="progress-bar" style="width:0%"></div></div>

    <!-- Last result -->
    <div id="last-result" class="hidden" style="margin-bottom:14px">
      <div class="card" style="padding:16px">
        <div class="card-title" style="font-size:0.85rem;margin-bottom:10px">📨 Last Response</div>
        <div id="result-content"></div>
      </div>
    </div>

    <!-- Action picker -->
    <div class="card">
      <div class="card-title">What do you want to do?</div>
      <div class="card-sub">Pick an action, fill in the details, then hit Send.</div>
      <div class="action-grid">
        <div class="action-btn selected" id="act-query_data" onclick="pickAction('query_data')">
          <span class="action-icon">📈</span>
          <div><div class="action-name">Look up a metric</div><div class="action-hint">Check a number like revenue or churn rate</div></div>
        </div>
        <div class="action-btn" id="act-analyze_trend" onclick="pickAction('analyze_trend')">
          <span class="action-icon">📉</span>
          <div><div class="action-name">Analyze a trend</div><div class="action-hint">See how a metric changed over time</div></div>
        </div>
        <div class="action-btn" id="act-consult_stakeholder" onclick="pickAction('consult_stakeholder')">
          <span class="action-icon">💬</span>
          <div><div class="action-name">Ask a colleague</div><div class="action-hint">Get advice from analyst, CEO, or risk officer</div></div>
        </div>
        <div class="action-btn" id="act-simulate_counterfactual" onclick="pickAction('simulate_counterfactual')">
          <span class="action-icon">🔬</span>
          <div><div class="action-name">Run a simulation</div><div class="action-hint">Test what would happen with a decision</div></div>
        </div>
        <div class="action-btn" id="act-make_decision" onclick="pickAction('make_decision')">
          <span class="action-icon">✅</span>
          <div><div class="action-name">Make final decision</div><div class="action-hint">Submit your conclusion and end the episode</div></div>
        </div>
      </div>

      <!-- Param fields per action -->
      <div id="params-query_data">
        <div class="param-field">
          <label>Which metric do you want to check?</label>
          <select id="p-metric">
            <option value="revenue">💰 Revenue</option>
            <option value="monthly_active_users">👥 Monthly Active Users</option>
            <option value="churn_rate">📉 Churn Rate</option>
            <option value="ad_spend">📢 Ad Spend</option>
            <option value="cac">🎯 Customer Acquisition Cost (CAC)</option>
            <option value="ltv">💎 Lifetime Value (LTV)</option>
          </select>
        </div>
      </div>

      <div id="params-analyze_trend" class="hidden">
        <div class="param-field">
          <label>Which metric to analyze?</label>
          <select id="p-trend-metric">
            <option value="revenue">💰 Revenue</option>
            <option value="monthly_active_users">👥 Monthly Active Users</option>
            <option value="churn_rate">📉 Churn Rate</option>
            <option value="ad_spend">📢 Ad Spend</option>
            <option value="cac">🎯 CAC</option>
            <option value="ltv">💎 LTV</option>
          </select>
        </div>
        <div class="param-field">
          <label>How many quarters back?</label>
          <select id="p-quarters">
            <option value="2">2 quarters</option>
            <option value="3">3 quarters</option>
            <option value="4">4 quarters</option>
          </select>
        </div>
      </div>

      <div id="params-consult_stakeholder" class="hidden">
        <div class="param-field">
          <label>Who do you want to talk to?</label>
          <select id="p-stakeholder">
            <option value="analyst">📊 Analyst — gives data-driven advice</option>
            <option value="ceo">👔 CEO — wants growth at all costs</option>
            <option value="risk_officer">🛡️ Risk Officer — cautious and conservative</option>
          </select>
        </div>
      </div>

      <div id="params-simulate_counterfactual" class="hidden">
        <div class="param-field">
          <label>What decision do you want to test?</label>
          <input type="text" id="p-decision" placeholder="e.g. increase ad spend by 20%"/>
        </div>
      </div>

      <div id="params-make_decision" class="hidden">
        <div class="param-field">
          <label>What is your final decision?</label>
          <input type="text" id="p-final-decision" placeholder="e.g. reduce churn by improving onboarding"/>
        </div>
        <div class="param-field">
          <label>Explain your reasoning (the more detail, the better your score)</label>
          <textarea id="p-explanation" placeholder="Based on the data I looked at, the main issue is... I consulted the analyst who said... Therefore I recommend..."></textarea>
        </div>
      </div>

      <button class="btn btn-primary" id="send-btn" onclick="sendAction()" style="margin-top:8px">Send Action</button>
      <button class="btn btn-secondary" onclick="restartGame()">↩ Start Over</button>
    </div>
  </div>

  <!-- ── SCREEN 3: Episode done ── -->
  <div id="screen-3" class="hidden">
    <div class="card" style="text-align:center">
      <div style="font-size:2.5rem;margin-bottom:8px" id="end-emoji">🎉</div>
      <div class="card-title" style="font-size:1.1rem" id="end-title">Episode Complete!</div>
      <div class="card-sub" id="end-sub" style="margin-bottom:18px">Here's how you did</div>
      <div class="score-card">
        <div class="score-num" id="final-score">—</div>
        <div class="score-label">Final Score (out of 100)</div>
      </div>
      <div id="end-stats" style="margin-top:14px"></div>
      <button class="btn btn-primary" onclick="restartGame()" style="margin-top:20px">🔄 Play Again</button>
      <button class="btn btn-secondary" onclick="changeDifficulty()">🎯 Try Different Difficulty</button>
    </div>
  </div>

</div><!-- /wizard -->

<div class="toast" id="toast"></div>

<script>
let selectedDiff = 'easy';
let selectedAction = 'query_data';
let curStep = 0;
let maxStep = 10;
let totalReward = 0;
let ws = null;
let pendingResolve = null;

// WebSocket helpers — all game state lives in one persistent WS connection
function openWS() {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const sock = new WebSocket(proto + '://' + location.host + '/ws');
    sock.onopen  = () => { ws = sock; resolve(); };
    sock.onerror = () => reject(new Error('WebSocket failed'));
    sock.onmessage = (e) => { if (pendingResolve) { const fn = pendingResolve; pendingResolve = null; fn(JSON.parse(e.data)); } };
    sock.onclose = () => { ws = null; };
  });
}

function wsSend(msg) {
  return new Promise((resolve, reject) => {
    if (!ws || ws.readyState !== 1) { reject(new Error('Not connected')); return; }
    pendingResolve = resolve;
    ws.send(JSON.stringify(msg));
  });
}

// Parse the nested response from WS — openenv wraps observation inside data.observation or data
function parseObs(resp) {
  if (resp && resp.data) {
    if (resp.data.observation) return { obs: resp.data.observation, reward: resp.data.reward || 0, done: resp.data.done || false };
    return { obs: resp.data, reward: resp.data.reward || 0, done: resp.data.done || false };
  }
  return { obs: {}, reward: 0, done: false };
}

function pickDiff(d) {
  selectedDiff = d;
  ['easy','medium','hard'].forEach(x => document.getElementById('diff-'+x).classList.toggle('selected', x===d));
}

function pickAction(a) {
  selectedAction = a;
  ['query_data','analyze_trend','consult_stakeholder','simulate_counterfactual','make_decision'].forEach(x => {
    document.getElementById('act-'+x).classList.toggle('selected', x===a);
    document.getElementById('params-'+x).classList.toggle('hidden', x!==a);
  });
}

function showToast(msg, type) {
  type = type || 'success';
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + type;
  setTimeout(function(){ t.className='toast'; }, 2800);
}

function setScreen(n) {
  [1,2,3].forEach(function(i){ document.getElementById('screen-'+i).classList.toggle('hidden', i!==n); });
  [1,2,3].forEach(function(i){
    document.getElementById('dot-'+i).className = 'step-dot ' + (i<n?'done':i===n?'active':'pending');
    if(i<3) document.getElementById('line-'+i).className = 'step-line'+(i<n?' done':'');
  });
}

async function startGame() {
  const btn = event.currentTarget;
  btn.disabled = true; btn.textContent = '⏳ Connecting...';
  try {
    // Close stale connection before opening a fresh one
    if (ws) { try { ws.close(); } catch(e){} ws = null; }
    await openWS();
    btn.textContent = '⏳ Starting...';
    const resp = await wsSend({type:'reset', data:{difficulty: selectedDiff, seed: 0}});
    const {obs} = parseObs(resp);
    curStep  = obs.step_count || 0;
    maxStep  = (obs.metadata && obs.metadata.max_steps) || (selectedDiff==='easy'?10:selectedDiff==='medium'?20:30);
    totalReward = 0;
    document.getElementById('objective-text').textContent = (obs.metadata && obs.metadata.objective) || 'Investigate the company and make a strategic decision.';
    document.getElementById('cur-step').textContent = curStep;
    document.getElementById('max-step').textContent  = maxStep;
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('last-result').classList.add('hidden');
    setScreen(2);
    pickAction('query_data');
    showToast('Episode started! Good luck 🚀');
  } catch(e) { showToast('Failed to connect: '+e.message, 'error'); }
  btn.disabled = false; btn.textContent = '▶ Start Episode';
}

function buildAction() {
  const params = {};
  if(selectedAction === 'query_data') {
    params.metric = document.getElementById('p-metric').value;
  } else if(selectedAction === 'analyze_trend') {
    params.metric = document.getElementById('p-trend-metric').value;
    params.quarters = parseInt(document.getElementById('p-quarters').value);
  } else if(selectedAction === 'consult_stakeholder') {
    params.stakeholder = document.getElementById('p-stakeholder').value;
  } else if(selectedAction === 'simulate_counterfactual') {
    params.decision = document.getElementById('p-decision').value || 'optimize_growth';
    params.parameters = {};
  } else if(selectedAction === 'make_decision') {
    params.decision = document.getElementById('p-final-decision').value || 'balanced_approach';
    params.parameters = {};
    params.explanation = document.getElementById('p-explanation').value || 'Based on available data.';
  }
  return {action_type: selectedAction, parameters: params};
}

function renderResult(obs, reward) {
  let html = '';
  if(obs.data_tables && Object.keys(obs.data_tables).filter(function(k){return k!=='quarter';}).length) {
    html += '<div class="metric-grid">';
    for(const [k,v] of Object.entries(obs.data_tables)) {
      if(k==='quarter') continue;
      const fmt = (v===null||v===undefined) ? 'N/A' : (typeof v==='number'?(v>1000?v.toLocaleString('en',{maximumFractionDigits:0}):(+v).toFixed(3)):String(v));
      html += '<div class="metric-tile"><div class="mname">'+k.replace(/_/g,' ')+'</div><div class="mval">'+fmt+'</div></div>';
    }
    html += '</div>';
  }
  if(obs.stakeholder_feedback) {
    html += '<div class="feedback-box" style="margin-top:10px"><div class="feedback-label">💬 Colleague says</div>'+obs.stakeholder_feedback+'</div>';
  }
  if(obs.simulation_results && Object.keys(obs.simulation_results).length) {
    html += '<div class="feedback-box" style="margin-top:10px"><div class="feedback-label">🔬 Simulation result</div>';
    for(const [k,v] of Object.entries(obs.simulation_results)) {
      html += '<div style="margin-top:4px"><span style="color:#8b949e">'+k+':</span> <b>'+(typeof v==='number'?v.toFixed(3):String(v))+'</b></div>';
    }
    html += '</div>';
  }
  const rCls = reward>0?'green':reward<0?'red':'blue';
  const rStr = (reward>=0?'+':'')+reward.toFixed(4);
  html += '<div class="result-row" style="margin-top:10px"><span class="result-label">Reward earned</span><span class="result-val '+rCls+'">'+rStr+'</span></div>';
  if(obs.metadata && obs.metadata.error) {
    html += '<div style="color:#f85149;font-size:0.82rem;margin-top:8px">⚠️ '+obs.metadata.error+'</div>';
  }
  if(!html) html = '<p style="color:#8b949e">No data returned for this action.</p>';
  document.getElementById('result-content').innerHTML = html;
  document.getElementById('last-result').classList.remove('hidden');
}

async function sendAction() {
  const btn = document.getElementById('send-btn');
  btn.disabled = true; btn.textContent = '⏳ Sending...';
  try {
    const action = buildAction();
    const resp = await wsSend({type:'step', data: action});
    const {obs, reward, done} = parseObs(resp);
    totalReward += reward;
    curStep = obs.step_count || curStep + 1;
    document.getElementById('cur-step').textContent = curStep;
    document.getElementById('progress-bar').style.width = Math.min(100,(curStep/maxStep)*100)+'%';
    renderResult(obs, reward);
    if(done) {
      const score = (obs.metadata && obs.metadata.final_score != null) ? obs.metadata.final_score : reward;
      setTimeout(function(){ showEndScreen(score, curStep); }, 600);
    } else {
      const labels = {query_data:'Metric queried ✓',analyze_trend:'Trend analyzed ✓',consult_stakeholder:'Colleague consulted ✓',simulate_counterfactual:'Simulation complete ✓',make_decision:'Decision submitted ✓'};
      showToast(labels[action.action_type] || 'Action sent ✓');
    }
  } catch(e) { showToast('Error: '+e.message, 'error'); }
  btn.disabled = false; btn.textContent = 'Send Action';
}

function showEndScreen(score, steps) {
  const pct = Math.round(score * 100);
  document.getElementById('final-score').textContent = pct;
  const emoji = pct>=80?'🏆':pct>=50?'👍':'💪';
  document.getElementById('end-emoji').textContent = emoji;
  document.getElementById('end-title').textContent = pct>=80?'Excellent work!':pct>=50?'Good job!':'Keep practicing!';
  document.getElementById('end-sub').textContent   = pct>=80?'You nailed it as CDO!':pct>=50?'Solid decision-making.':'Every episode teaches you more.';
  const rCls = totalReward>=0?'green':'red';
  const rStr = (totalReward>=0?'+':'')+totalReward.toFixed(3);
  document.getElementById('end-stats').innerHTML =
    '<div class="result-row"><span class="result-label">Difficulty</span><span class="result-val">'+selectedDiff+'</span></div>'+
    '<div class="result-row"><span class="result-label">Steps taken</span><span class="result-val">'+steps+' / '+maxStep+'</span></div>'+
    '<div class="result-row"><span class="result-label">Total reward</span><span class="result-val '+rCls+'">'+rStr+'</span></div>';
  if(ws) { try { ws.close(); } catch(e){} ws = null; }
  setScreen(3);
}

function restartGame() { setScreen(1); }
function changeDifficulty() { setScreen(1); }
</script>
</body>
</html>"""


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
