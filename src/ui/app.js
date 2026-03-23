/**
 * Multi-Agent Dev System — Frontend Logic
 * Handles API calls, polling, and UI updates.
 */

const API_BASE_URL = 'http://localhost:8000';
let pollInterval = null;

// DOM Elements
const form = document.getElementById('run-form');
const issueInput = document.getElementById('issue-url');
const submitBtn = document.getElementById('submit-btn');
const monitorCard = document.getElementById('pipeline-monitor');
const terminal = document.getElementById('terminal-body');
const statusText = document.getElementById('pipeline-status-text');
const retryCountText = document.getElementById('retry-count');
const steps = document.querySelectorAll('.step');
const resultArea = document.getElementById('result-area');
const errorArea = document.getElementById('error-area');
const healthChips = {
    google: document.getElementById('check-google'),
    github: document.getElementById('check-github'),
    docker: document.getElementById('check-docker')
};

/**
 * Health Check Loop
 */
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        updateHealthChip(healthChips.google, data.google_api_key);
        updateHealthChip(healthChips.github, data.github_token);
        updateHealthChip(healthChips.docker, data.docker);
    } catch (e) {
        console.error('Health check failed:', e);
        Object.values(healthChips).forEach(chip => updateHealthChip(chip, false));
    }
}

function updateHealthChip(chip, isOk) {
    chip.classList.remove('loading');
    chip.classList.add(isOk ? 'status-ok' : 'status-error');
}

/**
 * Run Pipeline
 */
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = issueInput.value.trim();
    if (!url) return;

    // Demo Mode toggle
    if (url.toLowerCase().includes('demo')) {
        runDemo();
        return;
    }

    // Reset UI
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    monitorCard.classList.remove('hidden');
    resultArea.classList.add('hidden');
    errorArea.classList.add('hidden');
    terminal.innerHTML = '';
    addLog(`Pipeline triggered for: ${url}`, 'info');
    resetSteps();

    try {
        const response = await fetch(`${API_BASE_URL}/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ issue_url: url })
        });

        const data = await response.json();
        const runId = data.run_id;
        addLog(`Run started. ID: ${runId}`, 'info');

        // Start polling
        startPolling(runId);
    } catch (e) {
        addLog(`Error starting pipeline: ${e.message}`, 'error');
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
    }
});

/**
 * Polling Logic
 */
function startPolling(runId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/status/${runId}`);
            const data = await response.json();

            updateStatusUI(data);

            if (data.status === 'done' || data.status === 'failed') {
                stopPipeline(data);
            }
        } catch (e) {
            console.error('Polling error:', e);
        }
    }, 2000); // Poll every 2 seconds
}

function updateStatusUI(data) {
    statusText.innerText = data.status.charAt(0).toUpperCase() + data.status.slice(1);
    retryCountText.innerText = data.retry_count;

    // Update stepper
    let currentFound = false;
    steps.forEach(step => {
        const stepStatus = step.dataset.status;

        if (data.status === stepStatus || (data.status === 'retrying' && stepStatus === 'coding')) {
            step.classList.add('active');
            step.classList.remove('completed');
            currentFound = true;
            addLog(`Agent: ${step.querySelector('.step-label').innerText} is now active...`, 'agent');
        } else if (!currentFound && data.status !== 'started') {
            step.classList.add('completed');
            step.classList.remove('active');
        } else {
            step.classList.remove('active', 'completed');
        }
    });
}

function stopPipeline(data) {
    clearInterval(pollInterval);
    submitBtn.classList.remove('loading');
    submitBtn.disabled = false;

    if (data.status === 'done') {
        addLog('Pipeline completed successfully!', 'success');
        document.getElementById('fix-explanation').innerText = "The pipeline has autonomously generated a fix, verified it with tests, and opened a pull request.";
        document.getElementById('pr-link').href = data.pr_url;
        resultArea.classList.remove('hidden');
    } else {
        addLog(`Pipeline failed: ${data.failure_reason}`, 'error');
        document.getElementById('failure-reason').innerText = data.failure_reason || 'Unknown error occurred.';
        errorArea.classList.remove('hidden');
    }
}

/**
 * Helpers
 */
function addLog(message, type) {
    const p = document.createElement('p');
    p.className = `log-line ${type}`;
    p.innerText = `> ${message}`;
    terminal.appendChild(p);
    terminal.scrollTop = terminal.scrollHeight;
}

function resetSteps() {
    steps.forEach(step => step.classList.remove('active', 'completed'));
}

/**
 * DEMO MODE LOGIC
 * Simulates a full agent pipeline run.
 */
async function runDemo() {
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    monitorCard.classList.remove('hidden');
    resultArea.classList.add('hidden');
    errorArea.classList.add('hidden');
    terminal.innerHTML = '';

    updateHealthChip(healthChips.google, true);
    updateHealthChip(healthChips.github, true);
    updateHealthChip(healthChips.docker, true);

    addLog('Demo mode activated. Simulating pipeline...', 'info');

    const sequence = [
        { status: 'researching', log: 'Research Agent: Scanning repository for "divide_by_zero" keywords...', delay: 2000 },
        { status: 'researching', log: 'Research Agent: Found src/math_utils.py. Identifying suspected function...', delay: 2000 },
        { status: 'coding', log: 'Coder Agent: Analyzing math_utils.py. Writing minimal fix...', delay: 3000 },
        { status: 'coding', log: 'Coder Agent: Fix generated (unified diff). Explaining changes...', delay: 2000 },
        { status: 'testing', log: 'Tester Agent: Generating pytest scenarios for edge cases...', delay: 3000 },
        { status: 'testing', log: 'Tester Agent: Running tests in Docker sandbox (no-network mode)...', delay: 4000 },
        { status: 'testing', log: 'Tester Agent: 3/3 tests passed! Validation successful.', delay: 2000 },
        { status: 'writing_pr', log: 'PR Writer Agent: Creating branch "fix/issue-123"...', delay: 2000 },
        { status: 'writing_pr', log: 'PR Writer Agent: Committing fix and opening Pull Request...', delay: 3000 },
        { status: 'done', log: 'Pipeline complete! PR opened on GitHub.', delay: 0 }
    ];

    for (const step of sequence) {
        if (step.status !== 'done') {
            updateStatusUI({ status: step.status, retry_count: 0 });
            addLog(step.log, step.status === 'researching' ? 'info' : 'agent');
            await new Promise(r => setTimeout(r, step.delay));
        }
    }

    stopPipeline({
        status: 'done',
        pr_url: 'https://github.com/octocat/hello-world/pull/42',
        failure_reason: null
    });
}

// Initialize
checkHealth();
setInterval(checkHealth, 5000); // Check health every 5s
