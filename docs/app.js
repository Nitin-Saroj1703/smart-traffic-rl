// Navigation and page routing
document.addEventListener('DOMContentLoaded', () => {
    // Menu toggle for mobile
    const menuBtn = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');
    
    if (menuBtn && sidebar) {
        menuBtn.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
    }

    // Page routing
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active from all links and pages
            navLinks.forEach(l => l.classList.remove('active'));
            pages.forEach(p => p.classList.remove('active'));
            
            // Add active to clicked link
            link.classList.add('active');
            
            // Show corresponding page
            const targetPageId = 'page-' + link.getAttribute('data-page');
            document.getElementById(targetPageId).classList.add('active');
            
            // Auto close sidebar on mobile
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('open');
            }

            // Render charts if needed when page becomes visible
            if (targetPageId === 'page-performance') renderPerformanceCharts();
            if (targetPageId === 'page-explainer') renderExplainerCharts();
            if (targetPageId === 'page-config') renderConfigChart();
        });
    });

    // Initialize Simulation Page
    initSimulation();

    // Initialize Config Sliders
    initConfigSliders();

    // Setup plot layouts
    const isDark = true;
    window.plotLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter', color: '#c9d1d9', size: 11 },
        margin: { l: 40, r: 20, t: 10, b: 25 },
        xaxis: { showgrid: false, zeroline: false, color: '#484f58' },
        yaxis: { showgrid: true, gridcolor: 'rgba(48,54,61,0.5)', zeroline: false, color: '#484f58' }
    };
});

// --- Simulation Logic (Mocked for static site) ---
let simInterval;
let simStep = 0;
const maxSimSteps = 3600;

function initSimulation() {
    renderGrid({0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0});
    
    const btnStart = document.getElementById('btn-start');
    const btnPause = document.getElementById('btn-pause');
    const btnReset = document.getElementById('btn-reset');
    const statusPill = document.getElementById('sim-status');
    const metricsDiv = document.getElementById('sim-metrics');

    btnStart.addEventListener('click', () => {
        statusPill.textContent = 'live';
        statusPill.className = 'status-pill pill-live';
        if (!simInterval) {
            simInterval = setInterval(() => stepSimulation(metricsDiv), 1500);
        }
    });

    btnPause.addEventListener('click', () => {
        statusPill.textContent = 'paused';
        statusPill.className = 'status-pill pill-paused';
        clearInterval(simInterval);
        simInterval = null;
    });

    btnReset.addEventListener('click', () => {
        statusPill.textContent = 'idle';
        statusPill.className = 'status-pill pill-idle';
        clearInterval(simInterval);
        simInterval = null;
        simStep = 0;
        renderGrid({0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0});
        metricsDiv.innerHTML = '<p class="start-hint">Hit <strong>Start</strong> to run the simulation.</p>';
        document.getElementById('trend-chart').innerHTML = ''; // clear chart
    });
}

function randInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1) + min);
}

// Data for live trend chart
let historyTimes = [];
let historyWait = [];
let historyCO2 = [];

function stepSimulation(metricsContainer) {
    simStep++;
    
    // Generate random phases for grid (0=NS G, 1=NS Y, 2=EW G, 3=EW Y)
    const phases = {};
    for (let i = 0; i < 9; i++) phases[i] = randInt(0, 3);
    renderGrid(phases);

    // Generate random metrics
    const waitTime = randInt(20, 100);
    const co2 = randInt(5000, 15000);
    const throughput = randInt(30, 80);
    const emergency = randInt(0, 2) > 1 ? 1 : 0;

    historyTimes.push(simStep);
    historyWait.push(waitTime);
    historyCO2.push(co2);
    
    if (historyTimes.length > 50) {
        historyTimes.shift();
        historyWait.shift();
        historyCO2.shift();
    }

    // Update metrics UI
    metricsContainer.innerHTML = `
        <div class="sim-metric">
            <div class="sim-metric-label">Vehicles waiting</div>
            <div class="sim-metric-value">${waitTime}</div>
        </div>
        <div class="sim-metric">
            <div class="sim-metric-label">CO2 output</div>
            <div class="sim-metric-value">${co2.toLocaleString()} mg/s</div>
        </div>
        <div class="sim-metric">
            <div class="sim-metric-label">Throughput</div>
            <div class="sim-metric-value">${throughput} veh/min</div>
        </div>
        <div class="sim-metric">
            <div class="sim-metric-label">Emergency vehicles</div>
            <div class="sim-metric-value">${emergency} active</div>
        </div>
        <div class="sim-step">Step ${simStep} / ${maxSimSteps}</div>
    `;

    // Render live trend chart
    if (historyTimes.length >= 2) {
        const traceWait = {
            x: historyTimes, y: historyWait,
            mode: 'lines', name: 'Wait',
            line: {color: '#58a6ff', width: 2},
            fill: 'tozeroy', fillcolor: 'rgba(88,166,255,0.06)'
        };
        const layout = Object.assign({}, window.plotLayout, {
            showlegend: false,
            margin: { l: 30, r: 10, t: 10, b: 20 },
            height: 180
        });
        Plotly.newPlot('trend-chart', [traceWait], layout, {displayModeBar: false});
    }
}

function renderGrid(phases) {
    const gridEl = document.getElementById('intersection-grid');
    if (!gridEl) return;
    
    let html = '';
    const cssClass = {0: 'phase-ns-green', 1: 'phase-ns-yellow', 2: 'phase-ew-green', 3: 'phase-ew-yellow'};
    const label = {0: 'NS', 1: 'NS', 2: 'EW', 3: 'EW'};
    
    for (let i = 0; i < 9; i++) {
        const p = phases[i];
        html += `<div class="grid-cell ${cssClass[p]}">I${i}<span class="grid-cell-sub">${label[p]}</span></div>`;
    }
    gridEl.innerHTML = html;
}

// --- Performance Charts ---
let performanceChartsRendered = false;
function renderPerformanceCharts() {
    if (performanceChartsRendered) return;
    
    const steps = Array.from({length: 50}, (_, i) => i + 1);
    
    // Generate dummy data similar to python script
    const baseWait = steps.map(() => 55 + (Math.random() - 0.5) * 16);
    const rlWait = steps.map(() => 30 + (Math.random() - 0.5) * 12);
    
    const baseCo2 = steps.map(() => 12000 + (Math.random() - 0.5) * 3000);
    const rlCo2 = steps.map(() => 8200 + (Math.random() - 0.5) * 2400);

    const layoutWait = Object.assign({}, window.plotLayout, {
        legend: {orientation: 'h', y: 1.15, font: {size: 10}},
        yaxis: {title: 'Seconds', showgrid: true, gridcolor: 'rgba(48,54,61,0.5)', color: '#484f58'}
    });
    
    Plotly.newPlot('wait-chart', [
        {x: steps, y: baseWait, name: 'Fixed-time', mode: 'lines', line: {color: '#f85149', width: 2}},
        {x: steps, y: rlWait, name: 'RL (MAPPO)', mode: 'lines', line: {color: '#58a6ff', width: 2}, fill: 'tonexty', fillcolor: 'rgba(88,166,255,0.06)'}
    ], layoutWait, {displayModeBar: false, responsive: true});

    const layoutCo2 = Object.assign({}, window.plotLayout, {
        legend: {orientation: 'h', y: 1.15, font: {size: 10}},
        yaxis: {title: 'mg/s', showgrid: true, gridcolor: 'rgba(48,54,61,0.5)', color: '#484f58'}
    });

    Plotly.newPlot('co2-chart', [
        {x: steps, y: baseCo2, name: 'Fixed-time', mode: 'lines', line: {color: '#f85149', width: 2}},
        {x: steps, y: rlCo2, name: 'RL (MAPPO)', mode: 'lines', line: {color: '#3fb950', width: 2}, fill: 'tonexty', fillcolor: 'rgba(63,185,80,0.06)'}
    ], layoutCo2, {displayModeBar: false, responsive: true});

    performanceChartsRendered = true;
}

// --- Explainer Charts ---
let explainerChartsRendered = false;
function renderExplainerCharts() {
    if (explainerChartsRendered) return;

    const features = ['Queue length', 'Wait time', 'Avg speed', 'CO2 level', 'Emergency', 'Phase duration', 'Neighbor queue', 'Time of day'].reverse();
    // Generate random importance sorted ascending for barh
    let importance = Array.from({length: 8}, () => Math.random() * 0.28 + 0.02).sort();
    
    const layoutFeat = Object.assign({}, window.plotLayout, {
        margin: {l: 100, r: 20, t: 10, b: 40},
        xaxis: {title: 'Weight', showgrid: true, gridcolor: 'rgba(48,54,61,0.5)', color: '#484f58'},
        yaxis: {showgrid: false, color: '#c9d1d9'}
    });
    
    Plotly.newPlot('feature-chart', [{
        type: 'bar',
        x: importance, y: features, orientation: 'h',
        marker: {
            color: importance,
            colorscale: [[0, '#58a6ff'], [1, '#f85149']]
        }
    }], layoutFeat, {displayModeBar: false, responsive: true});

    // Action probabilities
    const actions = ['Keep phase', 'NS to EW', 'EW to NS'];
    const probs = [0.6, 0.2, 0.2]; // Dummy dirichlet
    
    const layoutAct = Object.assign({}, window.plotLayout, {
        margin: {l: 50, r: 20, t: 20, b: 30},
        xaxis: {showgrid: false, color: '#c9d1d9'},
        yaxis: {title: 'Probability', range: [0, 1], showgrid: true, gridcolor: 'rgba(48,54,61,0.5)', color: '#484f58'}
    });

    Plotly.newPlot('action-chart', [{
        type: 'bar',
        x: actions, y: probs,
        marker: { color: ['#58a6ff', '#3fb950', '#d29922'] },
        text: probs.map(p => (p*100).toFixed(0) + '%'),
        textposition: 'outside',
        textfont: {color: '#8b949e', size: 11}
    }], layoutAct, {displayModeBar: false, responsive: true});

    explainerChartsRendered = true;
}

// --- Config Interaction ---
function initConfigSliders() {
    const ids = ['throughput', 'wait', 'emissions', 'emergency'];
    
    ids.forEach(id => {
        const slider = document.getElementById('sl-' + id);
        const valSpan = document.getElementById('val-' + id);
        
        slider.addEventListener('input', (e) => {
            valSpan.textContent = parseFloat(e.target.value).toFixed(id==='emergency'?0:2);
            renderConfigChart();
        });
    });

    document.getElementById('btn-save-config').addEventListener('click', () => {
        const msg = document.getElementById('config-saved');
        msg.style.display = 'block';
        setTimeout(() => { msg.style.display = 'none'; }, 3000);
    });
}

function renderConfigChart() {
    const v1 = parseFloat(document.getElementById('sl-throughput').value);
    const v2 = parseFloat(document.getElementById('sl-wait').value);
    const v3 = parseFloat(document.getElementById('sl-emissions').value);
    const v4 = parseFloat(document.getElementById('sl-emergency').value);

    const labels = ['Throughput', 'Wait penalty', 'Emissions', 'Emergency'];
    const values = [v1, v2, v3, v4];
    
    const layout = Object.assign({}, window.plotLayout, {
        margin: {l: 40, r: 20, t: 20, b: 30},
        xaxis: {showgrid: false, color: '#c9d1d9'},
        yaxis: {showgrid: true, gridcolor: 'rgba(48,54,61,0.5)', color: '#484f58'}
    });

    Plotly.newPlot('weights-chart', [{
        type: 'bar',
        x: labels, y: values,
        marker: { color: ['#58a6ff', '#f85149', '#3fb950', '#d29922'] },
        text: values.map(v => v.toFixed(2)),
        textposition: 'outside',
        textfont: {color: '#8b949e', size: 11}
    }], layout, {displayModeBar: false, responsive: true});
}
