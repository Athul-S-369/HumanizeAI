# Text Humanizer - Research Domain

A powerful web application that transforms AI-written content into natural, human-like text optimized for research and academic domains. The tool removes plagiarism markers and makes content appear 100% human-written.

## Features

- 🔄 **Intelligent Paraphrasing**: Uses advanced NLP techniques to replace words with synonyms and restructure sentences
- ✍️ **Natural Language Processing**: Removes common AI writing patterns and makes text sound authentically human
- 🔍 **Plagiarism Removal**: Transforms content to pass plagiarism detection tools
- 📚 **Research Focused**: Optimized specifically for academic and research domain content
- 🎨 **Modern UI**: Beautiful, responsive web interface

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server**:
   ```bash
   python app.py
   ```

   The server will start on `http://localhost:5000`

4. **Open the frontend**:
   - Open `static/index.html` in your web browser, or
   - Navigate to `http://localhost:5000/static/index.html`

## Usage

1. Paste your AI-written content into the input text area
2. Click the "Humanize Text" button
3. The humanized version will appear in the output area
4. Copy or download the result

## How It Works

The humanizer uses multiple techniques:

1. **Synonym Replacement**: Replaces words with appropriate synonyms using WordNet
2. **Sentence Restructuring**: Varies sentence length and structure
3. **Pattern Removal**: Replaces common AI writing patterns with natural alternatives
4. **Punctuation Variation**: Adds human-like punctuation variations
5. **Text Normalization**: Ensures proper spacing and capitalization

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **NLP Library**: NLTK (Natural Language Toolkit)
- **WordNet**: For synonym extraction

## API Endpoints

### POST `/api/humanize`
Humanizes the provided text.

**Request Body**:
```json
{
  "text": "Your AI-written text here"
}
```

**Response**:
```json
{
  "original": "Original text",
  "humanized": "Humanized text",
  "success": true
}
```

### GET `/api/health`
Health check endpoint.

## Notes

- The first run may take longer as NLTK downloads required data files
- Processing time depends on text length
- For best results, use with research/academic content
- The tool works best with complete sentences and paragraphs

## License

This project is open source and available for use.




