# Import necessary libraries
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.probability import FreqDist
from nltk.cluster.util import cosine_distance

text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California that designs, develops, and sells consumer electronics, computer software, and online services. The company's hardware products include the iPhone smartphone, the iPad tablet computer, the Mac personal computer, the iPod portable media player, the Apple smartwatch, and the Apple TV digital media player."

stop_words = set(stopwords.words("english"))
sents = sent_tokenize(text)
preprocess_sents = []
preprocess_words = []

def preprocess_text(sentence):
    # Remove punctuation characters
    sentence = re.sub(r"[^a-zA-Z0-9]", " ", sentence)

    # Tokenization
    words = word_tokenize(sentence)

    # Remove stopwords
    filtered_words = [w for w in words if w.lower() not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in filtered_words]

    return stemmed



print(preprocess_text(text))
for sent in sents:
    preprocess_sents.append(preprocess_text(sent))

print(preprocess_sents)

flat_preprocessed_words = [word for sentence in preprocess_sents for word in sentence]
print(flat_preprocessed_words)

print(FreqDist(flat_preprocessed_words))
print(FreqDist(sent))
