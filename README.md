# NLP Article Analyzer

A complete Natural Language Processing pipeline that scrapes web articles, preprocesses text using two different approaches, identifies topics through multiple techniques, and generates summaries — comparing methods at each stage to evaluate trade-offs.

## What it does

Give it a URL of any news article (English or Spanish) and the pipeline will:

1. **Scrape & clean** the HTML, stripping scripts, nav bars, footers, and other noise using BeautifulSoup
2. **Preprocess the text** using two methods side by side:
   - **spaCy** (lemmatization) — preserves word meaning ("running" → "run")
   - **NLTK** (stemming) — more aggressive reduction ("running" → "run-")
3. **Detect the topic** using three different techniques:
   - **TF-IDF** — statistical word importance scoring
   - **Sentence Embeddings** — semantic vector representation (384 dimensions)
   - **Zero-shot classification** — BART-based categorization without prior training
4. **Generate summaries** comparing:
   - **Abstractive** — DistilBART rewrites the content into a new summary
   - **Extractive** — selects the most relevant original sentences

The pipeline automatically detects the article's language and applies the correct models and stopwords.

## Results

Tested on 5 real-world articles across different domains and languages:

| Article | Detected Topic | Expected | Correct |
|---------|---------------|----------|---------|
| Nature - AlphaFold | technology | science | ~* |
| arXiv - MiniGPT | science | technology | ~* |
| El Mundo - Griezmann | sports | sports | ✓ |
| Simply Wall St - First Solar | business | business | ✓ |
| La Razón - Politics | infrastructure | politics | ✗ |

*\*Science/technology misclassifications are understandable given the overlap between these categories.*

**Key findings:**
- spaCy lemmatization produces more readable and semantically useful tokens than NLTK stemming
- Zero-shot classification achieves ~60% accuracy with generic categories — fine-tuned categories would improve this
- Abstractive summarization works better for English articles; extractive is more reliable for Spanish

## Tech Stack

- **Python 3.10+**
- **BeautifulSoup4** — HTML scraping and cleaning
- **spaCy** — tokenization, lemmatization, stopword removal
- **NLTK** — stemming, stopword removal (comparative method)
- **Sentence Transformers** — multilingual semantic embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- **HuggingFace Transformers** — zero-shot classification (`facebook/bart-large-mnli`) and summarization (`sshleifer/distilbart-cnn-12-6`)
- **Scikit-learn** — TF-IDF vectorization

## Setup

```bash
# Clone the repo
git clone https://github.com/KAMZ226L/nlp-article-analyzer.git
cd nlp-article-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

## Usage

```bash
python nlp_pipeline.py
```

The script processes 5 pre-configured articles and outputs the full analysis for each one: cleaned text stats, token counts for both preprocessing methods, detected topic with confidence scores, and two summaries.

To analyze your own articles, edit the `articles` list in `nlp_pipeline.py`:

```python
articles = [
    {
        "nom": "Your Article",
        "url": "https://example.com/article",
        "tema_esperat": "technology"
    }
]
```

## Project Structure

```
nlp-article-analyzer/
├── nlp_pipeline.py        # Main pipeline script
├── requirements.txt       # Python dependencies
├── README.md
└── LICENSE
```

## What I learned

- Lemmatization (spaCy) preserves more semantic information than stemming (NLTK), making it better for downstream NLP tasks
- Zero-shot classification is powerful for rapid prototyping but struggles with overlapping categories
- Multilingual NLP adds complexity — models trained primarily on English perform worse on Spanish text
- Combining multiple techniques (statistical + neural) gives a more complete picture than relying on a single approach

## License

MIT
