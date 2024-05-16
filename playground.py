# Import necessary libraries
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, subjectivity
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.probability import FreqDist
from nltk.cluster.util import cosine_distance
text = subjectivity.raw("plot.tok.gt9.5000")
#text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California that designs, develops, and sells consumer electronics, computer software, and online services. The company's hardware products include the iPhone smartphone, the iPad tablet computer, the Mac personal computer, the iPod portable media player, the Apple smartwatch, and the Apple TV digital media player."
#
stop_words = set(stopwords.words("english"))
sents = sent_tokenize(text)

preprocess_sents = []
preprocess_words = []

def preprocess_text(sentence):
    sentence = re.sub(r"[^a-zA-Z0-9]", " ", sentence)
    words = word_tokenize(sentence)
    filtered_words = [w for w in words if w.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in filtered_words]

    return stemmed


def score_sentences(sentences, word_freq):
    sentence_scores = {}

    for i, sentence in enumerate(sentences):
        for word in sentence:
            if word in word_freq:
                if i in sentence_scores:
                    sentence_scores[i] += word_freq[word]
                else:
                    sentence_scores[i] = word_freq[word]

    return sentence_scores

#print(preprocess_text(text))
for sent in sents:
    preprocess_sents.append(preprocess_text(sent))

#print(preprocess_sents)

flat_preprocessed_words = [word for sentence in preprocess_sents for word in sentence]
#print(flat_preprocessed_words)

#print(FreqDist(flat_preprocessed_words))
word_freq = FreqDist(flat_preprocessed_words)
print(word_freq)

sentence_scores = score_sentences(preprocess_sents, word_freq)
print(sentence_scores)

summary_sentences = []
if sentence_scores:
    sorted_scores = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    top_sentences = sorted_scores[:3]  # Select the top 3 sentences as the summary

    for index, _ in top_sentences:
        summary_sentences.append(sents[index])

summary = ' '.join(summary_sentences)
print("\nSummary:")
print(summary)
print(summary_sentences)
