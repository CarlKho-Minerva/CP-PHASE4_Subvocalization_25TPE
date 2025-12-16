/**
 * Typographic Watermarking - Content Script
 * Intercepts copy events and injects Unicode fingerprints
 */

// Unicode space fingerprints for each AI provider
const FINGERPRINTS = {
  // OpenAI
  'chatgpt.com': '\u2009',      // Thin Space
  'chat.openai.com': '\u2009',  // Thin Space

  // Anthropic
  'claude.ai': '\u200A',        // Hair Space

  // Google
  'gemini.google.com': '\u2005', // Four-Per-Em Space

  // Others
  'poe.com': '\u2004',          // Three-Per-Em Space
  'copilot.microsoft.com': '\u2006', // Six-Per-Em Space
  'perplexity.ai': '\u2007',    // Figure Space
  'www.perplexity.ai': '\u2007',
  'pi.ai': '\u2008',            // Punctuation Space
  'huggingface.co': '\u205F',   // Medium Mathematical Space
};

// Get fingerprint for current site
function getFingerprint() {
  const hostname = window.location.hostname;
  return FINGERPRINTS[hostname] || null;
}

// Get AI name for logging
function getAIName() {
  const hostname = window.location.hostname;
  if (hostname.includes('chatgpt') || hostname.includes('openai')) return 'ChatGPT';
  if (hostname.includes('claude')) return 'Claude';
  if (hostname.includes('gemini')) return 'Gemini';
  if (hostname.includes('poe')) return 'Poe';
  if (hostname.includes('copilot')) return 'Copilot';
  if (hostname.includes('perplexity')) return 'Perplexity';
  if (hostname.includes('pi.ai')) return 'Pi';
  if (hostname.includes('huggingface')) return 'HuggingChat';
  return 'Unknown AI';
}

// Inject watermark into text
function injectWatermark(text) {
  const fingerprint = getFingerprint();
  if (!fingerprint) return text;

  // Replace standard ASCII spaces with the fingerprint space
  return text.replace(/ /g, fingerprint);
}

// Listen for copy events
document.addEventListener('copy', (e) => {
  const fingerprint = getFingerprint();
  if (!fingerprint) return; // Not on a supported AI site

  // Get the selected text
  const selection = window.getSelection();
  if (!selection || selection.toString().trim() === '') return;

  const originalText = selection.toString();
  const watermarkedText = injectWatermark(originalText);

  // Override the clipboard data
  e.clipboardData.setData('text/plain', watermarkedText);
  e.preventDefault(); // Prevent the default copy

  console.log(`🔒 Typographic Watermark: Injected ${getAIName()} signature (U+${fingerprint.charCodeAt(0).toString(16).toUpperCase().padStart(4, '0')})`);
});

// Log that we're active
console.log(`🔒 Typographic Watermark active on ${getAIName()}`);
