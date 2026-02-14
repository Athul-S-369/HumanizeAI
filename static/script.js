const API_URL = `${window.location.origin}/api`;

const inputText = document.getElementById('input-text');
const outputText = document.getElementById('output-text');
const humanizeBtn = document.getElementById('humanize-btn');
const clearBtn = document.getElementById('clear-btn');
const copyBtn = document.getElementById('copy-btn');
const downloadBtn = document.getElementById('download-btn');
const stats = document.getElementById('stats');
const notification = document.getElementById('notification');

// Humanize text
humanizeBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();
    
    if (!text) {
        showNotification('Please enter some text to humanize', 'error');
        return;
    }
    
    // Disable button and show loading
    humanizeBtn.disabled = true;
    const btnText = humanizeBtn.querySelector('.btn-text');
    const btnLoader = humanizeBtn.querySelector('.btn-loader');
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';
    
    try {
        const response = await fetch(`${API_URL}/humanize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (data.success) {
            outputText.value = data.humanized;
            updateStats(text, data.humanized);
            showNotification('Text humanized successfully!');
        } else {
            showNotification(data.error || 'An error occurred', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Failed to connect to server. Make sure the backend is running.', 'error');
    } finally {
        // Re-enable button
        humanizeBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

// Clear both text areas
clearBtn.addEventListener('click', () => {
    inputText.value = '';
    outputText.value = '';
    stats.style.display = 'none';
});

// Copy to clipboard
copyBtn.addEventListener('click', async () => {
    const text = outputText.value;
    
    if (!text) {
        showNotification('No text to copy', 'error');
        return;
    }
    
    try {
        await navigator.clipboard.writeText(text);
        showNotification('Copied to clipboard!');
    } catch (error) {
        // Fallback for older browsers
        outputText.select();
        document.execCommand('copy');
        showNotification('Copied to clipboard!');
    }
});

// Download as text file
downloadBtn.addEventListener('click', () => {
    const text = outputText.value;
    
    if (!text) {
        showNotification('No text to download', 'error');
        return;
    }
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'humanized-text.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showNotification('File downloaded!');
});

// Calculate similarity and update stats
function updateStats(original, humanized) {
    const originalLength = original.length;
    const humanizedLength = humanized.length;
    
    // Simple similarity calculation (word overlap)
    const originalWords = new Set(original.toLowerCase().split(/\s+/));
    const humanizedWords = new Set(humanized.toLowerCase().split(/\s+/));
    
    const intersection = new Set([...originalWords].filter(x => humanizedWords.has(x)));
    const union = new Set([...originalWords, ...humanizedWords]);
    
    const similarity = Math.round((intersection.size / union.size) * 100);
    
    document.getElementById('original-length').textContent = originalLength.toLocaleString();
    document.getElementById('humanized-length').textContent = humanizedLength.toLocaleString();
    document.getElementById('similarity').textContent = similarity + '%';
    
    stats.style.display = 'flex';
}

// Show notification
function showNotification(message, type = 'success') {
    notification.textContent = message;
    notification.className = `notification ${type} show`;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

// Allow Enter+Ctrl/Cmd to humanize
inputText.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        humanizeBtn.click();
    }
});

