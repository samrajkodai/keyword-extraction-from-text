import nltk
nltk.download('punkt')
nltk.download('stopwords')

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
# nltk.download('punkt')
# nltk.download('stopwords')


# Open and read the text file and analyze it
text_file = open("npl.txt")
text = text_file.read()
print(type(text))
print(text)
print(len(text))
print("\n")

print("Sentences tokenizing")
# Sentence tokenizing
sentences = sent_tokenize(text)
print(len(sentences))
print("sentences")

# Word tokenizing
print("Word tokenizing")
words = word_tokenize(text)
print(len(words))
print(words)
print("\n")

# Frequency Distribution
print("Frequency Distribution")
fDist = FreqDist(words)
fDist.most_common(10)
print(fDist)

# Plot the frequency Graph
print("Plot the frequency Graph")
print("\n")
fDist.plot(10)
plt.show()

# Remove punctuation marks
print("Remove punctuation marks")

words_no_punch = []
for w in words:
    if w.isalpha():
        words_no_punch.append(w.lower())

print(words_no_punch)
print(len(words_no_punch))

# Plotting graph without punctuation marks
print("Plotting graph without punctuation marks")
fDist = FreqDist(words_no_punch)
fDist.most_common(10)
print(fDist)
plt.show()
print("\n")
# List of stopwords
print("List of stopwords")
stopwords = stopwords.words("english")
print(stopwords)
print(len(stopwords))
print("\n")
# List of stopwords
print("Removing stopwords")

clean_words = []
for w in words_no_punch:
    if w not in stopwords:
        clean_words.append(w)

print(clean_words)
print(len(clean_words))

# Final Frequency Distribution
print("Final Frequency Distribution")
fDist = FreqDist(clean_words)
fDist.most_common(10)
print(fDist)

# Plot the final frequency Graph
print("Plot the final frequency Graph")
print("\n")
fDist.plot(10)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
n=cv.fit_transform(clean_words).toarray()
