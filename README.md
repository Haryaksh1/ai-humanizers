# 🧠 AI Text Humanizer

Welcome to **AI Text Humanizer** : an open-source project that transforms robotic or AI-generated writing into fluent, natural, human-like text.

**Goal for Hacktoberfest 2025:** build a full-scale, industry-level humanizer through community contributions. 🚀

---

## Overview

This project contains two Python humanizers:

- 🟢 **Balanced** : preserves meaning and prioritizes grammar & readability.  
- 🔴 **Aggressive** : stronger rephrasing and stylistic variation (more creative).

Both use modern NLP tools (Transformers, spaCy, TextStat, NLTK) to refine grammar and tone.

---

## Quick start

1. Clone the repo:
```bash
git clone https://github.com/Haryaksh1/AI-Text-Humanizer.git
cd AI-Text-Humanizer
```
2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## CLI Usage

The CLI provides a simple command-line interface for humanizing text:

### Basic Usage
```bash
# Humanize text directly
python cli/humanize_cli.py --model balanced --input "Furthermore, it is important to note that artificial intelligence is transforming industries."

# Use aggressive model for stronger rephrasing
python cli/humanize_cli.py --model aggressive --input "Moreover, the comprehensive analysis demonstrates significant improvements."

# Save output to file
python cli/humanize_cli.py --model balanced --input "Your AI text here" --output humanized.txt

# Process text from file
python cli/humanize_cli.py --model aggressive --file input.txt --output output.txt

# Show detailed statistics
python cli/humanize_cli.py --model balanced --input "Your text" --stats
```

### CLI Options
- `--model`: Choose humanizer model (`balanced` or `aggressive`)
  - **balanced**: Preserves meaning, focuses on readability
  - **aggressive**: Stronger rephrasing, more creative transformations
- `--input`: Input text to humanize (use quotes for multi-word text)
- `--file`: Input file containing text to humanize
- `--output`: Output file to save humanized text (optional, prints to stdout if not specified)
- `--stats`: Show humanization statistics and metrics
- `--quiet`: Suppress progress messages

## API Usage

The Flask API provides RESTful endpoints for integrating humanizers into other applications:

### Starting the API Server
```bash
python api/app.py
```
The API will be available at `http://localhost:5000`

### API Endpoints

#### POST /humanize
Humanize AI-generated text using balanced or aggressive models.

**Request:**
```json
{
  "model": "balanced",
  "text": "Furthermore, it is important to note that artificial intelligence is transforming industries.",
  "include_stats": true
}
```

**Response:**
```json
{
  "humanized_text": "Also, AI is really changing industries in a big way.",
  "model_used": "balanced",
  "processing_time": 0.234,
  "success": true,
  "stats": {
    "initial_ai_score": 2.1,
    "final_ai_score": 0.3,
    "improvement_percentage": 85.7,
    "target_achieved": true,
    "input_length": 89,
    "output_length": 52,
    "quality_metrics": {
      "overall_quality": 78.5,
      "readability_score": 82.3,
      "grade_level": 8.2
    }
  }
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-22 14:30:15",
  "service": "AI Text Humanizer API"
}
```

#### GET /
API documentation with examples and endpoint details.

### cURL Examples
```bash
# Basic humanization
curl -X POST http://localhost:5000/humanize \
  -H "Content-Type: application/json" \
  -d '{"model": "balanced", "text": "Furthermore, it is important to note that..."}'

# With statistics
curl -X POST http://localhost:5000/humanize \
  -H "Content-Type: application/json" \
  -d '{"model": "aggressive", "text": "Moreover, the comprehensive analysis demonstrates...", "include_stats": true}'

# Health check
curl http://localhost:5000/health
```

### Python Integration Example
```python
import requests

# Humanize text via API
response = requests.post('http://localhost:5000/humanize', json={
    'model': 'balanced',
    'text': 'Your AI-generated text here',
    'include_stats': True
})

result = response.json()
print("Humanized:", result['humanized_text'])
print("Improvement:", result['stats']['improvement_percentage'], "%")
```
## Tech stack

- Python

- Transformers (Hugging Face)

- spaCy

- NLTK

- TextStat

- scikit-learn

- pandas / numpy

## Contributing

We welcome all contributions — small fixes or large features. This repo is beginner-friendly.  

### How to contribute

- Fork the repository.  

- Create a branch: git checkout -b feature-name.  

- Commit changes: git commit -m "Add <feature>".  

- Push and open a pull request.  

Please open an issue first if you plan to add a new feature (see “How to open a new issue” below).  

## How to open a new issue

We encourage issues for:

- New features (UI, new model variant, API)

- Model improvements (grammar, readability)

- Bug reports / performance optimizations

- Research & evaluation

Steps:

1.Go to the Issues tab: https://github.com/Haryaksh1/AI-Text-Humanizer/issues

2.Click New issue.

3.Use the template (title, short description, steps/files changed).

4.Add labels: hacktoberfest, good first issue, python (you can pick them on the right).

5.Comment “I’d like to work on this” and wait to be assigned.

## Hacktoberfest 2025

This repo participates in Hacktoberfest 2025. Look for issues labelled:  

- hacktoberfest  

- good first issue  

- python  

If your PR is merged or marked hacktoberfest-accepted by a maintainer, it will count.  

## License

MIT License 

## Maintainer

Maintained by @Haryaksh1
If you need help, tag me in the issue or PR and I’ll reply quickly.

