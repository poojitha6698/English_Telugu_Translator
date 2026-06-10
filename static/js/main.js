const inputText = document.getElementById('inputText');
const translateBtn = document.getElementById('translateBtn');
const btnText = document.getElementById('btnText');
const btnSpinner = document.getElementById('btnSpinner');
const outputArea = document.getElementById('outputArea');
const charCount = document.getElementById('charCount');
const clearBtn = document.getElementById('clearBtn');
const copyBtn = document.getElementById('copyBtn');
const copyMsg = document.getElementById('copyMsg');
const errorBox = document.getElementById('errorBox');

inputText.addEventListener('input', () => {
  charCount.textContent = inputText.value.length;
});

clearBtn.addEventListener('click', () => {
  inputText.value = '';
  charCount.textContent = '0';
  resetOutput();
  hideError();
});

copyBtn.addEventListener('click', () => {
  const text = outputArea.dataset.translation;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    copyMsg.classList.remove('hidden');
    setTimeout(() => copyMsg.classList.add('hidden'), 2000);
  });
});

document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    inputText.value = chip.dataset.text;
    charCount.textContent = inputText.value.length;
    doTranslate();
  });
});

translateBtn.addEventListener('click', doTranslate);

inputText.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) doTranslate();
});

function setLoading(loading) {
  translateBtn.disabled = loading;
  btnText.textContent = loading ? 'Translating...' : 'Translate';
  btnSpinner.classList.toggle('hidden', !loading);
}

function resetOutput() {
  outputArea.innerHTML = '<span class="placeholder-text">Translation will appear here...</span>';
  delete outputArea.dataset.translation;
}

function hideError() {
  errorBox.classList.add('hidden');
  errorBox.textContent = '';
}

async function doTranslate() {
  const text = inputText.value.trim();
  if (!text) return;

  setLoading(true);
  hideError();
  resetOutput();

  try {
    const res = await fetch('/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || 'Translation failed');
    }

    outputArea.textContent = data.translated;
    outputArea.dataset.translation = data.translated;
  } catch (err) {
    errorBox.textContent = `⚠ ${err.message}`;
    errorBox.classList.remove('hidden');
    resetOutput();
  } finally {
    setLoading(false);
  }
}
