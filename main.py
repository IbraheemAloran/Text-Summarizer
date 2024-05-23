import summarize_mod as sum
import nltk
from nltk.corpus import movie_reviews

text = movie_reviews.raw("pos\cv000_29590.txt")
print(sum.summarize(text, 0.2))

