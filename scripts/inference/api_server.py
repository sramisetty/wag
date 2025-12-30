"""
WAG Ad Copy API Server
======================
Flask API server for integrating WAG ad copy generation with applications.

Usage:
    python api_server.py                    # Start server on port 5000
    python api_server.py --port 8080        # Start on custom port
    python api_server.py --host 0.0.0.0     # Allow external connections

API Endpoints:
    POST /api/generate          - Generate ad copy
    POST /api/generate/batch    - Generate for multiple items
    GET  /api/health            - Health check
    GET  /api/models            - List available models

Author: Enterprise Architecture Team
Created: December 2025
"""

import os
import json
import logging
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from functools import wraps
import time

# Try to import markdown, fallback to basic rendering
try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation."""
    nav_html = get_nav_html('home')
    base_styles = get_base_styles()

    page_styles = '''
        .endpoint { background: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #3498db; }
        .method { display: inline-block; padding: 3px 8px; border-radius: 4px; font-weight: bold; margin-right: 10px; }
        .get { background: #27ae60; color: white; }
        .post { background: #3498db; color: white; }
        .try-it { margin-top: 20px; }
        textarea { width: 100%; height: 150px; font-family: monospace; padding: 10px; border-radius: 5px; border: 1px solid #bdc3c7; }
        button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #2980b9; }
        #result { margin-top: 15px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #2c3e50; }
        select { padding: 8px 12px; border-radius: 5px; border: 1px solid #bdc3c7; font-size: 14px; min-width: 250px; }
        .model-info { font-size: 12px; color: #7f8c8d; margin-top: 5px; }
        .controls { display: flex; gap: 15px; align-items: center; flex-wrap: wrap; margin-top: 15px; }
        .temperature-group { display: flex; align-items: center; gap: 10px; }
        .temperature-group input { width: 60px; padding: 8px; border-radius: 5px; border: 1px solid #bdc3c7; }
        .status { padding: 5px 10px; border-radius: 4px; font-size: 12px; }
        .status.loading { background: #f39c12; color: white; }
        .status.ready { background: #27ae60; color: white; }
        .status.error { background: #e74c3c; color: white; }
        .status-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        @media (max-width: 900px) { .status-panels { grid-template-columns: 1fr; } }
        .training-panel { background: white; padding: 20px; border-radius: 10px; border-left: 4px solid #9b59b6; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        .training-panel h3 { margin-top: 0; color: #9b59b6; }
        .gpu-panel { background: white; padding: 20px; border-radius: 10px; border-left: 4px solid #e67e22; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        .gpu-panel h3 { margin-top: 0; color: #e67e22; }
        .gpu-card { background: #f8f9fa; border-radius: 8px; padding: 15px; margin-top: 10px; }
        .gpu-name { font-weight: bold; color: #2c3e50; margin-bottom: 10px; font-size: 14px; }
        .gpu-bars { display: flex; flex-direction: column; gap: 8px; }
        .gpu-bar-container { display: flex; align-items: center; gap: 10px; }
        .gpu-bar-label { font-size: 11px; color: #7f8c8d; width: 60px; }
        .gpu-bar { flex: 1; background: #ecf0f1; border-radius: 4px; height: 16px; overflow: hidden; }
        .gpu-bar-fill { height: 100%; transition: width 0.3s ease; border-radius: 4px; }
        .gpu-bar-fill.util { background: linear-gradient(90deg, #27ae60, #2ecc71); }
        .gpu-bar-fill.mem { background: linear-gradient(90deg, #e67e22, #f39c12); }
        .gpu-bar-fill.temp { background: linear-gradient(90deg, #3498db, #e74c3c); }
        .gpu-bar-value { font-size: 11px; color: #2c3e50; width: 55px; text-align: right; }
        .gpu-unavailable { color: #7f8c8d; font-style: italic; }
        .progress-bar { background: #ecf0f1; border-radius: 10px; height: 20px; overflow: hidden; margin: 10px 0; }
        .progress-fill { background: linear-gradient(90deg, #9b59b6, #3498db); height: 100%; transition: width 0.3s ease; }
        .training-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-top: 15px; }
        .stat-box { background: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; }
        .stat-value { font-size: 16px; font-weight: bold; color: #2c3e50; }
        .stat-label { font-size: 11px; color: #7f8c8d; }
        .training-idle { color: #7f8c8d; font-style: italic; }
        .training-active { color: #27ae60; }
        .training-complete { color: #3498db; }
        .api-section { background: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    '''

    return '''
<!DOCTYPE html>
<html>
<head>
    <title>WAG Ad Copy API</title>
    <style>
        ''' + base_styles + page_styles + '''
    </style>
</head>
<body>
    ''' + nav_html + '''
    <div class="container">
        <h1>WAG Ad Copy Generation API</h1>
        <p>Generate retail advertising headlines and body copy using AI.</p>

        <div class="status-panels">
            <div class="training-panel" id="trainingPanel">
                <h3>Fine-Tuning Status</h3>
                <div id="trainingContent">
                    <p class="training-idle">Loading training status...</p>
                </div>
            </div>

            <div class="gpu-panel" id="gpuPanel">
                <h3>GPU Status</h3>
                <div id="gpuContent">
                    <p class="training-idle">Loading GPU status...</p>
                </div>
            </div>
        </div>

    <h2>Endpoints</h2>

    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/health</code>
        <p>Health check - verify the API is running</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/models</code>
        <p>List available AI models</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/generate</code>
        <p>Generate ad copy for products</p>
        <pre>{
  "products": [{"description": "ADVIL PM 20CT", "brand": "Advil"}],
  "price": "$8.99",
  "offer": "BOGO 50%"
}</pre>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/generate/batch</code>
        <p>Generate ad copy for multiple items</p>
    </div>

    <h2>Try It</h2>
    <div class="try-it">
        <div class="form-group">
            <label for="model">Model</label>
            <select id="model">
                <option value="">Loading models...</option>
            </select>
            <span id="modelStatus" class="status loading">Loading...</span>
            <div class="model-info">Select an Ollama model to use for generation</div>
        </div>

        <div class="form-group">
            <label for="input">Request Body</label>
            <textarea id="input">{
  "products": [{"description": "ADVIL PM 20CT", "brand": "Advil"}],
  "price": "$8.99",
  "offer": "BOGO 50%"
}</textarea>
        </div>

        <div class="controls">
            <button onclick="generate()">Generate Ad Copy</button>
            <div class="temperature-group">
                <label for="temperature">Temperature:</label>
                <input type="number" id="temperature" value="0.7" min="0" max="2" step="0.1">
            </div>
            <button onclick="loadModels()" style="background: #95a5a6;">Refresh Models</button>
        </div>

        <pre id="result">Results will appear here...</pre>
    </div>

    <script>
    let defaultModel = '';

    async function loadModels() {
        const select = document.getElementById('model');
        const status = document.getElementById('modelStatus');
        status.textContent = 'Loading...';
        status.className = 'status loading';

        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            select.innerHTML = '';
            defaultModel = data.default || '';

            if (data.models && data.models.length > 0) {
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model + (model === defaultModel ? ' (default)' : '');
                    if (model === defaultModel) {
                        option.selected = true;
                    }
                    select.appendChild(option);
                });
                status.textContent = data.models.length + ' models available';
                status.className = 'status ready';
            } else {
                select.innerHTML = '<option value="">No models found</option>';
                status.textContent = 'No models';
                status.className = 'status error';
            }
        } catch (e) {
            select.innerHTML = '<option value="">Error loading models</option>';
            status.textContent = 'Error: ' + e.message;
            status.className = 'status error';
        }
    }

    async function generate() {
        const input = document.getElementById('input').value;
        const model = document.getElementById('model').value;
        const temperature = parseFloat(document.getElementById('temperature').value) || 0.7;
        const result = document.getElementById('result');

        result.textContent = 'Generating...';

        try {
            // Parse input and add model/temperature
            let requestBody = JSON.parse(input);
            if (model) {
                requestBody.model = model;
            }
            requestBody.temperature = temperature;

            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(requestBody)
            });
            const data = await response.json();
            result.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
            result.textContent = 'Error: ' + e.message;
        }
    }

    async function loadTrainingStatus() {
        const content = document.getElementById('trainingContent');

        try {
            const response = await fetch('/api/training/status');
            const data = await response.json();

            if (data.status === 'idle' || data.status === 'error') {
                content.innerHTML = `<p class="training-idle">${data.message || 'No training in progress'}</p>`;
            } else if (data.status === 'completed') {
                content.innerHTML = `
                    <p class="training-complete"><strong>Training Completed!</strong></p>
                    <div class="progress-bar"><div class="progress-fill" style="width: 100%"></div></div>
                    <div class="training-stats">
                        <div class="stat-box">
                            <div class="stat-value">${data.current_step}</div>
                            <div class="stat-label">Total Steps</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${data.total_epochs}</div>
                            <div class="stat-label">Epochs</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${data.loss ? data.loss.toFixed(4) : 'N/A'}</div>
                            <div class="stat-label">Final Loss</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${data.elapsed_formatted || 'N/A'}</div>
                            <div class="stat-label">Total Time</div>
                        </div>
                    </div>
                `;
            } else {
                // Training in progress
                const percent = data.percent_complete || 0;
                content.innerHTML = `
                    <p class="training-active"><strong>Training in Progress...</strong> ${data.phase || ''}</p>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${percent}%"></div>
                    </div>
                    <p style="text-align: center; margin: 5px 0;">${percent.toFixed(1)}% Complete</p>
                    <div class="training-stats">
                        <div class="stat-box">
                            <div class="stat-value">${data.current_step} / ${data.total_steps}</div>
                            <div class="stat-label">Steps</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${data.current_epoch ? data.current_epoch.toFixed(2) : 0} / ${data.total_epochs}</div>
                            <div class="stat-label">Epochs</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${data.loss ? data.loss.toFixed(4) : 'N/A'}</div>
                            <div class="stat-label">Loss</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${data.elapsed_formatted || '00:00:00'}</div>
                            <div class="stat-label">Elapsed</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${data.eta_formatted || 'Calculating...'}</div>
                            <div class="stat-label">ETA</div>
                        </div>
                    </div>
                `;
            }
        } catch (e) {
            content.innerHTML = `<p class="training-idle">Unable to fetch training status</p>`;
        }
    }

    async function loadGpuStatus() {
        const content = document.getElementById('gpuContent');

        try {
            const response = await fetch('/api/gpu/status');
            const data = await response.json();

            if (!data.available) {
                content.innerHTML = `<p class="gpu-unavailable">GPU monitoring unavailable: ${data.error || 'No GPU detected'}</p>`;
                return;
            }

            let html = '';
            for (const gpu of data.gpus) {
                const tempColor = gpu.temperature_c > 80 ? '#e74c3c' : (gpu.temperature_c > 60 ? '#f39c12' : '#27ae60');
                const tempPercent = gpu.temperature_c ? Math.min((gpu.temperature_c / 100) * 100, 100) : 0;

                html += `
                    <div class="gpu-card">
                        <div class="gpu-name">GPU ${gpu.index}: ${gpu.name}</div>
                        <div class="gpu-bars">
                            <div class="gpu-bar-container">
                                <span class="gpu-bar-label">Utilization</span>
                                <div class="gpu-bar">
                                    <div class="gpu-bar-fill util" style="width: ${gpu.gpu_utilization}%"></div>
                                </div>
                                <span class="gpu-bar-value">${gpu.gpu_utilization}%</span>
                            </div>
                            <div class="gpu-bar-container">
                                <span class="gpu-bar-label">Memory</span>
                                <div class="gpu-bar">
                                    <div class="gpu-bar-fill mem" style="width: ${gpu.memory_percent}%"></div>
                                </div>
                                <span class="gpu-bar-value">${(gpu.memory_used_mb/1024).toFixed(1)}/${(gpu.memory_total_mb/1024).toFixed(0)}GB</span>
                            </div>
                            ${gpu.temperature_c ? `
                            <div class="gpu-bar-container">
                                <span class="gpu-bar-label">Temp</span>
                                <div class="gpu-bar">
                                    <div class="gpu-bar-fill temp" style="width: ${tempPercent}%; background: ${tempColor}"></div>
                                </div>
                                <span class="gpu-bar-value">${gpu.temperature_c}°C</span>
                            </div>
                            ` : ''}
                            ${gpu.power_draw_w ? `
                            <div class="gpu-bar-container">
                                <span class="gpu-bar-label">Power</span>
                                <div class="gpu-bar">
                                    <div class="gpu-bar-fill util" style="width: ${(gpu.power_draw_w/gpu.power_limit_w*100).toFixed(0)}%"></div>
                                </div>
                                <span class="gpu-bar-value">${gpu.power_draw_w.toFixed(0)}/${gpu.power_limit_w.toFixed(0)}W</span>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }

            content.innerHTML = html || '<p class="gpu-unavailable">No GPUs found</p>';
        } catch (e) {
            content.innerHTML = `<p class="gpu-unavailable">Error loading GPU status</p>`;
        }
    }

    // Load models, training status, and GPU status on page load
    document.addEventListener('DOMContentLoaded', () => {
        loadModels();
        loadTrainingStatus();
        loadGpuStatus();
        // Refresh status every 5 seconds
        setInterval(loadTrainingStatus, 5000);
        setInterval(loadGpuStatus, 3000);
    });
    </script>
    </div>
</body>
</html>
'''

# Configuration
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.getenv('WAG_MODEL', 'wag-copywriter')
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '60'))

# Training status file path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_STATUS_FILE = os.path.join(SCRIPT_DIR, '..', 'output', 'training_status.json')

# Documentation file paths
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
DOCS = {
    'readme': os.path.join(SCRIPT_DIR, '..', 'README.md'),
    'model-guide': os.path.join(PROJECT_ROOT, 'MODEL_GUIDE.md'),
    'overview': os.path.join(PROJECT_ROOT, 'SCRIPTS_OVERVIEW.md'),
    'plan': os.path.join(PROJECT_ROOT, 'LLM_FINETUNING_PLAN.md'),
}


def render_markdown(text):
    """Render markdown to HTML."""
    if HAS_MARKDOWN:
        # Use comprehensive extensions for better rendering
        extensions = [
            'tables',
            'fenced_code',
            'codehilite',
            'toc',
            'nl2br',
            'sane_lists',
            'smarty',
        ]
        extension_configs = {
            'codehilite': {
                'css_class': 'highlight',
                'guess_lang': False,
            }
        }
        try:
            return markdown.markdown(
                text,
                extensions=extensions,
                extension_configs=extension_configs
            )
        except Exception:
            # Fallback to basic extensions if some aren't available
            return markdown.markdown(text, extensions=['tables', 'fenced_code'])
    else:
        # Basic fallback rendering
        html = text

        # Code blocks first (before other processing)
        html = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)

        # Headers (must match start of line)
        html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Horizontal rules
        html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Inline code
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

        # Links
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

        # Lists (basic)
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(<li>.*</li>\n?)+', r'<ul>\g<0></ul>', html)

        # Tables (basic)
        def convert_table(match):
            lines = match.group(0).strip().split('\n')
            if len(lines) < 2:
                return match.group(0)

            result = '<table>'
            for i, line in enumerate(lines):
                if '---' in line:
                    continue
                cells = [c.strip() for c in line.strip('|').split('|')]
                tag = 'th' if i == 0 else 'td'
                row = ''.join(f'<{tag}>{c}</{tag}>' for c in cells)
                result += f'<tr>{row}</tr>'
            result += '</table>'
            return result

        html = re.sub(r'(\|.+\|\n)+', convert_table, html)

        # Paragraphs
        paragraphs = html.split('\n\n')
        processed = []
        for p in paragraphs:
            p = p.strip()
            if p and not p.startswith('<'):
                p = f'<p>{p}</p>'
            processed.append(p)
        html = '\n'.join(processed)

        return html


def get_nav_html(active=''):
    """Generate navigation HTML."""
    nav_items = [
        ('/', 'Home', 'home'),
        ('/docs/readme', 'README', 'readme'),
        ('/docs/model-guide', 'Model Guide', 'model-guide'),
        ('/docs/overview', 'Overview', 'overview'),
        ('/docs/plan', 'Fine-Tuning Plan', 'plan'),
    ]

    items_html = ''
    for href, label, key in nav_items:
        active_class = ' active' if key == active else ''
        items_html += f'<a href="{href}" class="nav-item{active_class}">{label}</a>'

    return f'''
    <nav class="main-nav">
        <div class="nav-brand">WAG Ad Copy</div>
        <div class="nav-links">{items_html}</div>
    </nav>
    '''


def get_base_styles():
    """Get base CSS styles."""
    return '''
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
               margin: 0; padding: 0; background: #f5f6fa; color: #2c3e50; line-height: 1.7; }
        .main-nav { background: #2c3e50; padding: 0 20px; display: flex; align-items: center;
                    justify-content: space-between; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    position: sticky; top: 0; z-index: 100; }
        .nav-brand { color: white; font-size: 20px; font-weight: bold; padding: 15px 0; }
        .nav-links { display: flex; gap: 5px; }
        .nav-item { color: #bdc3c7; text-decoration: none; padding: 15px 15px; transition: all 0.2s; }
        .nav-item:hover { color: white; background: #34495e; }
        .nav-item.active { color: white; background: #3498db; }
        .container { max-width: 1000px; margin: 0 auto; padding: 30px 20px; }
        .content-card { background: white; border-radius: 10px; padding: 30px 40px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-top: 0; font-size: 28px; }
        h2 { color: #34495e; margin-top: 35px; padding-bottom: 8px; border-bottom: 2px solid #ecf0f1; font-size: 22px; }
        h3 { color: #2c3e50; margin-top: 25px; font-size: 18px; }
        h4 { color: #7f8c8d; margin-top: 20px; font-size: 16px; }
        p { margin: 12px 0; }
        pre { background: #1e2838; color: #e8e8e8; padding: 16px 20px; border-radius: 8px;
              overflow-x: auto; font-size: 13px; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
              line-height: 1.5; margin: 15px 0; }
        code { background: #e8f4f8; color: #c7254e; padding: 2px 6px; border-radius: 3px;
               font-size: 13px; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; }
        pre code { background: none; color: inherit; padding: 0; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }
        th, td { border: 1px solid #dfe6e9; padding: 12px 15px; text-align: left; }
        th { background: #3498db; color: white; font-weight: 600; }
        tr:nth-child(even) { background: #f8f9fa; }
        tr:hover { background: #eef5ff; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
        ul, ol { padding-left: 30px; margin: 15px 0; }
        li { margin: 8px 0; }
        li > ul, li > ol { margin: 5px 0; }
        .doc-meta { color: #7f8c8d; font-size: 13px; margin-bottom: 25px; padding: 12px 15px;
                    background: #f8f9fa; border-radius: 5px; border-left: 3px solid #3498db; }
        blockquote { border-left: 4px solid #3498db; margin: 20px 0; padding: 15px 25px;
                     background: #f8f9fa; color: #555; font-style: italic; }
        blockquote p { margin: 0; }
        hr { border: none; border-top: 2px solid #ecf0f1; margin: 35px 0; }
        .highlight { background: #1e2838; border-radius: 8px; }
        img { max-width: 100%; height: auto; }
        strong { color: #2c3e50; }
        em { color: #555; }

        /* Checkbox styling for task lists */
        input[type="checkbox"] { margin-right: 8px; }

        /* Responsive */
        @media (max-width: 768px) {
            .content-card { padding: 20px; }
            .nav-links { flex-wrap: wrap; }
            .nav-item { padding: 10px; font-size: 14px; }
            pre { font-size: 12px; padding: 12px; }
            table { font-size: 12px; }
            th, td { padding: 8px 10px; }
        }
    '''


def log_request(f):
    """Decorator to log API requests."""
    @wraps(f)
    def decorated(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{request.method} {request.path} - {elapsed:.2f}s")
        return result
    return decorated


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        response = requests.get(f'{OLLAMA_URL}/api/tags', timeout=5)
        ollama_status = 'healthy' if response.status_code == 200 else 'unhealthy'
    except:
        ollama_status = 'unreachable'

    return jsonify({
        'status': 'healthy',
        'ollama': ollama_status,
        'model': DEFAULT_MODEL,
        'version': '1.0.0'
    })


@app.route('/api/training/status', methods=['GET'])
def training_status():
    """Get current training status."""
    try:
        if os.path.exists(TRAINING_STATUS_FILE):
            with open(TRAINING_STATUS_FILE, 'r') as f:
                status = json.load(f)

            # Calculate human-readable elapsed time
            elapsed = status.get('elapsed_seconds', 0)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            status['elapsed_formatted'] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

            # Calculate ETA
            eta = status.get('estimated_remaining_seconds')
            if eta:
                hours, remainder = divmod(eta, 3600)
                minutes, seconds = divmod(remainder, 60)
                status['eta_formatted'] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            else:
                status['eta_formatted'] = None

            return jsonify(status)
        else:
            return jsonify({
                'status': 'idle',
                'phase': 'not_started',
                'message': 'No training in progress. Run train.py to start.',
                'percent_complete': 0
            })
    except Exception as e:
        logger.error(f"Error reading training status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/docs/<doc_name>', methods=['GET'])
def show_documentation(doc_name):
    """Render documentation pages."""
    if doc_name not in DOCS:
        return "Documentation not found", 404

    doc_path = DOCS[doc_name]
    if not os.path.exists(doc_path):
        return f"File not found: {doc_path}", 404

    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get file modification time
        import datetime
        mtime = os.path.getmtime(doc_path)
        modified = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')

        # Render markdown to HTML
        html_content = render_markdown(content)

        # Document titles
        titles = {
            'readme': 'README - Scripts Documentation',
            'model-guide': 'Model Guide',
            'overview': 'Scripts Overview',
            'plan': 'LLM Fine-Tuning Plan',
        }

        return f'''
<!DOCTYPE html>
<html>
<head>
    <title>{titles.get(doc_name, doc_name)} - WAG Ad Copy</title>
    <style>{get_base_styles()}</style>
</head>
<body>
    {get_nav_html(doc_name)}
    <div class="container">
        <div class="content-card">
            <div class="doc-meta">
                Last modified: {modified} | File: {os.path.basename(doc_path)}
            </div>
            {html_content}
        </div>
    </div>
</body>
</html>
'''
    except Exception as e:
        logger.error(f"Error rendering documentation: {e}")
        return f"Error loading documentation: {str(e)}", 500


@app.route('/api/gpu/status', methods=['GET'])
def gpu_status():
    """Get GPU utilization and memory usage."""
    import subprocess

    try:
        # Query nvidia-smi for GPU stats
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode != 0:
            return jsonify({'error': 'nvidia-smi failed', 'available': False}), 500

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    memory_used = float(parts[3])
                    memory_total = float(parts[4])
                    memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0

                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'gpu_utilization': float(parts[2]),
                        'memory_used_mb': memory_used,
                        'memory_total_mb': memory_total,
                        'memory_percent': round(memory_percent, 1),
                        'temperature_c': float(parts[5]) if parts[5] != '[N/A]' else None,
                        'power_draw_w': float(parts[6]) if parts[6] != '[N/A]' else None,
                        'power_limit_w': float(parts[7]) if parts[7] != '[N/A]' else None,
                    })

        return jsonify({
            'available': True,
            'gpu_count': len(gpus),
            'gpus': gpus
        })

    except FileNotFoundError:
        return jsonify({'error': 'nvidia-smi not found', 'available': False})
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'nvidia-smi timeout', 'available': False}), 500
    except Exception as e:
        logger.error(f"GPU status error: {e}")
        return jsonify({'error': str(e), 'available': False}), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available Ollama models."""
    try:
        response = requests.get(f'{OLLAMA_URL}/api/tags', timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return jsonify({
                'models': [m['name'] for m in models],
                'default': DEFAULT_MODEL
            })
    except Exception as e:
        logger.error(f"Error listing models: {e}")

    return jsonify({'error': 'Could not retrieve models'}), 500


@app.route('/api/generate', methods=['POST'])
@log_request
def generate_copy():
    """
    Generate ad copy for given products/offers.

    Request body:
    {
        "products": [
            {"description": "ADVIL PM 20CT", "brand": "Advil"}
        ],
        "price": "$8.99",
        "offer": "BOGO 50%",
        "limit": "2",
        "temperature": 0.7,
        "model": "wag-copywriter"  // optional
    }

    Response:
    {
        "headline": "Advil Pain Relief",
        "body_copy": "Select varieties. Limit 2.",
        "raw_response": "...",
        "model": "wag-copywriter",
        "elapsed_ms": 1234
    }
    """
    start_time = time.time()

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400

        # Build prompt
        prompt = build_prompt(data)

        # Get model
        model = data.get('model', DEFAULT_MODEL)
        temperature = data.get('temperature', 0.7)

        # Call Ollama
        response = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': temperature
                }
            },
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            return jsonify({'error': 'Model generation failed'}), 500

        raw_response = response.json().get('response', '')
        headline, body_copy = parse_response(raw_response)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return jsonify({
            'headline': headline,
            'body_copy': body_copy,
            'raw_response': raw_response,
            'model': model,
            'elapsed_ms': elapsed_ms
        })

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/batch', methods=['POST'])
@log_request
def generate_batch():
    """
    Generate ad copy for multiple items.

    Request body:
    {
        "items": [
            {
                "products": [{"description": "ADVIL PM 20CT", "brand": "Advil"}],
                "price": "$8.99",
                "offer": "BOGO 50%"
            },
            ...
        ],
        "temperature": 0.7,
        "model": "wag-copywriter"
    }

    Response:
    {
        "results": [
            {"headline": "...", "body_copy": "...", "index": 0},
            ...
        ],
        "total": 2,
        "elapsed_ms": 5000
    }
    """
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or 'items' not in data:
            return jsonify({'error': 'items array required'}), 400

        items = data['items']
        model = data.get('model', DEFAULT_MODEL)
        temperature = data.get('temperature', 0.7)

        results = []
        for i, item in enumerate(items):
            prompt = build_prompt(item)

            response = requests.post(
                f'{OLLAMA_URL}/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': temperature}
                },
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                raw = response.json().get('response', '')
                headline, body_copy = parse_response(raw)
                results.append({
                    'index': i,
                    'headline': headline,
                    'body_copy': body_copy,
                    'success': True
                })
            else:
                results.append({
                    'index': i,
                    'error': 'Generation failed',
                    'success': False
                })

        elapsed_ms = int((time.time() - start_time) * 1000)

        return jsonify({
            'results': results,
            'total': len(items),
            'successful': sum(1 for r in results if r.get('success')),
            'elapsed_ms': elapsed_ms
        })

    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/eab', methods=['POST'])
@log_request
def generate_from_eab():
    """
    Process EAB data and generate copy for ad slots.

    Request body:
    {
        "slots": [
            {
                "page": 1,
                "layout": "A",
                "wic_codes": ["691500", "691501"],
                "price": "$9.99"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        if not data or 'slots' not in data:
            return jsonify({'error': 'slots array required'}), 400

        results = []
        for slot in data['slots']:
            # Build products from WIC codes (simplified - just use codes)
            products = [{'description': f'WIC: {wic}'} for wic in slot.get('wic_codes', [])]

            item = {
                'products': products,
                'price': slot.get('price'),
                'offer': slot.get('offer'),
                'limit': slot.get('limit')
            }

            prompt = build_prompt(item)

            response = requests.post(
                f'{OLLAMA_URL}/api/generate',
                json={
                    'model': DEFAULT_MODEL,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                raw = response.json().get('response', '')
                headline, body_copy = parse_response(raw)
                results.append({
                    'page': slot.get('page'),
                    'layout': slot.get('layout'),
                    'headline': headline,
                    'body_copy': body_copy,
                    'success': True
                })
            else:
                results.append({
                    'page': slot.get('page'),
                    'layout': slot.get('layout'),
                    'error': 'Generation failed',
                    'success': False
                })

        return jsonify({
            'results': results,
            'total': len(data['slots'])
        })

    except Exception as e:
        logger.error(f"EAB generation error: {e}")
        return jsonify({'error': str(e)}), 500


def build_prompt(data):
    """Build generation prompt from request data."""
    parts = ["Generate a headline and body copy for this retail advertisement.", ""]

    products = data.get('products', [])
    if products:
        parts.append("Products:")
        for p in products[:5]:
            desc = p.get('description', 'Unknown product')
            brand = p.get('brand', '')
            if brand:
                parts.append(f"  - {desc} (Brand: {brand})")
            else:
                parts.append(f"  - {desc}")
        parts.append("")

    if data.get('price'):
        parts.append(f"Price: {data['price']}")
    if data.get('offer'):
        parts.append(f"Offer: {data['offer']}")
    if data.get('limit'):
        parts.append(f"Limit: {data['limit']}")

    return '\n'.join(parts)


def parse_response(text):
    """Parse model response into headline and body copy."""
    import re

    headline = ""
    body_copy = ""

    # Extract headline
    headline_match = re.search(r'Headline:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if headline_match:
        headline = headline_match.group(1).strip().strip('"')

    # Extract body copy
    body_match = re.search(r'BodyCopy:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if body_match:
        body_copy = body_match.group(1).strip().strip('"')

    # Fallback
    if not headline and text:
        lines = text.strip().split('\n')
        if lines:
            headline = lines[0].strip()
            headline = re.sub(r'^Headline:\s*', '', headline, flags=re.IGNORECASE).strip('"')
            if len(lines) > 1:
                body_copy = lines[1].strip()
                body_copy = re.sub(r'^BodyCopy:\s*', '', body_copy, flags=re.IGNORECASE).strip('"')

    return headline, body_copy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='WAG Ad Copy API Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              WAG Ad Copy Generation API Server               ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    POST /api/generate        - Generate ad copy              ║
║    POST /api/generate/batch  - Batch generation              ║
║    POST /api/generate/eab    - Process EAB slots             ║
║    GET  /api/health          - Health check                  ║
║    GET  /api/models          - List models                   ║
╠══════════════════════════════════════════════════════════════╣
║  Server: http://{args.host}:{args.port}                            ║
║  Model:  {DEFAULT_MODEL:<40}        ║
╚══════════════════════════════════════════════════════════════╝
""")

    app.run(host=args.host, port=args.port, debug=args.debug)
