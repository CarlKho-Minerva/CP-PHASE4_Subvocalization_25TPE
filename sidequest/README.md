# 🔒 Typographic Watermarking

**Invisible AI Text Attribution via Unicode Substitution**

A Chrome extension that embeds invisible "fingerprints" into AI-generated text by replacing standard ASCII spaces with visually-identical Unicode variants.

## The Idea

Every time you copy text from an AI chat (ChatGPT, Claude, Gemini, etc.), this extension automatically replaces the standard space character (U+0020) with a model-specific Unicode space that looks identical but can be detected programmatically.

| AI Model | Unicode Space | Character |
|----------|---------------|-----------|
| ChatGPT | U+2009 | Thin Space |
| Claude | U+200A | Hair Space |
| Gemini | U+2005 | Four-Per-Em Space |
| Copilot | U+2006 | Six-Per-Em Space |
| Perplexity | U+2007 | Figure Space |
| Poe | U+2004 | Three-Per-Em Space |

## Installation

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (top right toggle)
3. Click **Load unpacked**
4. Select the `chrome-extension` folder

## Usage

1. Go to any supported AI chat (ChatGPT, Claude, Gemini, etc.)
2. Copy any text (Cmd+C / Ctrl+C)
3. The watermark is automatically injected!
4. Open `decoder.html` to detect the source of any text

## Demo

Open the browser console on any AI chat site to see:
```
🔒 Typographic Watermark active on ChatGPT
```

When you copy text:
```
🔒 Typographic Watermark: Injected ChatGPT signature (U+2009)
```

## Limitations

- The watermark can be stripped by:
  - Pasting into Notepad (Windows)
  - Some IDEs that normalize whitespace
  - Text sanitization scripts
- However, it survives in:
  - Google Docs
  - Microsoft Word
  - Most social media platforms
  - Email clients

## For the arXiv Paper

This project demonstrates **post-hoc typographic watermarking** as an alternative to logit-level watermarking. Key research questions:

1. **Robustness**: Which platforms preserve Unicode spaces?
2. **Detectability**: Can users notice the substitution?
3. **Scalability**: Can we encode more information (model version, timestamp)?

## License

MIT
