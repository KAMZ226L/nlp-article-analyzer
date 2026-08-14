
# PRACTICA  PLN


from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import requests
import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('stopwords')
nltk.download('punkt')


# CARREGAR MODELS

print("Carregant models...")

# Models spaCy
try:
    nlp_es = spacy.load("es_core_news_sm")
except:
    nlp_es = None

try:
    nlp_en = spacy.load("en_core_web_sm")
except:
    nlp_en = None

embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

print("Models carregats.\n")


#  Processar HTML
 
def descargar_html(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        return r.text if r.status_code == 200 else None
    except:
        return None

def netejar_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    texto = soup.get_text(separator=" ")
    return " ".join(texto.split())


# DETECCIÓ IDIOMA

def detectar_idioma(texto):
    palabras_es = ['el', 'la', 'de', 'que', 'en', 'y', 'los', 'las']
    palabras_en = ['the', 'of', 'and', 'to', 'in', 'for']
    
    texto_lower = texto[:1000].lower()
    count_es = sum(texto_lower.count(f' {p} ') for p in palabras_es)
    count_en = sum(texto_lower.count(f' {p} ') for p in palabras_en)
    
    return 'spanish' if count_es > count_en else 'english'


# Preprocessament (2 versions)


def preprocesar_v1_spacy(texto):
    """Versió 1: spaCy"""
    idioma = detectar_idioma(texto)
    nlp = nlp_en if idioma == 'english' and nlp_en else nlp_es
    
    doc = nlp(texto[:100000])
    tokens = [token.lemma_.lower() for token in doc 
              if token.is_alpha and not token.is_stop and len(token.text) > 2]
    return tokens

def preprocesar_v2_nltk(texto):
    """Versió 2: NLTK"""
    idioma = detectar_idioma(texto)
    idioma_nltk = 'english' if idioma == 'english' else 'spanish'
    
    tokens = texto.lower().split()
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
    
    stop_words = set(stopwords.words(idioma_nltk))
    tokens = [t for t in tokens if t not in stop_words]
    
    stemmer = SnowballStemmer(idioma_nltk)
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


# Identificar tema


def generar_tfidf(tokens):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([" ".join(tokens)])
    return X, vectorizer

def generar_embedding(texto):
    return embedder.encode(texto[:1000])

def detectar_tema(texto):
    labels = ["sports", "medicine", "infrastructure", "business", "science", "technology", "politics"]
    result = classifier(texto[:1000], labels)
    return result


#  Generar resums 

def resumen_v1_model(texto):
    """Versió 1: Model"""
    texto_corto = " ".join(texto.split()[:400])
    try:
        result = summarizer(texto_corto, max_length=130, min_length=30, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Error: {e}"

def resumen_v2_extractiu(texto):
    """Versió 2: Extractiu"""
    frases = [s.strip() + '.' for s in texto.split('.') if len(s.strip()) > 30]
    return " ".join(frases[:3]) if frases else texto[:200]


# 5 ARTICLES 

articles = [
    {"nom": "Nature - AlphaFold", "url": "https://www.nature.com/articles/s41586-024-07487-w", "tema_esperat": "science"},
    {"nom": "arXiv - MiniGPT", "url": "https://arxiv.org/abs/2304.10592", "tema_esperat": "technology"},
    {"nom": "El Mundo - Griezmann", "url": "https://www.elmundo.es/deportes/futbol/champions-league/2025/09/30/68dc348afc6c833b218b4598.html", "tema_esperat": "sports"},
    {"nom": "First Solar", "url": "https://simplywall.st/es/stocks/us/semiconductors/nasdaq-fslr/first-solar/news/first-solar-fslr-cae-un-65por-ciento-tras-los-detalles-de-te", "tema_esperat": "business"},
    {"nom": "Pedro sanchez ", "url": "https://www.larazon.es/espana/sanchez-planta-europa-recibira-eurodiputados-que-evaluan-avances-corrupcion_202602156991af719243cc133c49d4be.html", "tema_esperat": "politics"}
]


# PIPELINE


resultats = []

for i, art in enumerate(articles):

    print(f"ARTICLE {i+1}: {art['nom']}")
    print(f"{'='*60}")
    
    html = descargar_html(art['url'])
    if not html:
        continue
    
    # PAS 1
    texto = netejar_html(html)
    print(f"Pas 1: {len(texto)} caràcters")
    
    # PAS 2 - COMPARATIVA
    print("\n-- Pas 2: Preprocessament --")
    tokens_v1 = preprocesar_v1_spacy(texto)
    tokens_v2 = preprocesar_v2_nltk(texto)
    print(f"  V1 (spaCy): {len(tokens_v1)} tokens - {tokens_v1[:5]}")
    print(f"  V2 (NLTK): {len(tokens_v2)} tokens - {tokens_v2[:5]}")
    
    tokens = tokens_v1
    
    # PAS 3
    print("\n-- Pas 3: Tema --")
    X_tfidf, _ = generar_tfidf(tokens)
    print(f"  TF-IDF: {X_tfidf.shape}")
    
    embedding = generar_embedding(texto)
    print(f"  Embedding: {embedding.shape}")
    
    tema_result = detectar_tema(texto)
    tema = tema_result['labels'][0]
    print(f"  Tema: {tema} (esperat: {art['tema_esperat']})")
    
    # PAS 4 - COMPARATIVA
    print("\n-- Pas 4: Resums --")
    resum1 = resumen_v1_model(texto)
    print(f"  V1 (model): {resum1[:100]}...")
    
    resum2 = resumen_v2_extractiu(texto)
    print(f"  V2 (extractiu): {resum2[:100]}...")
    
    resultats.append({
        "nom": art['nom'],
        "tema_esperat": art['tema_esperat'],
        "tema_detectat": tema,
        "tokens_v1": len(tokens_v1),
        "tokens_v2": len(tokens_v2)
    })

# RESUM


print("RESUM")
print(f"{'='*50}")

for res in resultats:
    print(f"{res['nom']}: {res['tema_detectat']} ({res['tokens_v1']}/{res['tokens_v2']} tokens)")

print("\n Completat!")