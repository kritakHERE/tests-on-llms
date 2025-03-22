import nltk
import torch
import torch.nn.functional as F
from nltk.util import ngrams
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Sample text corpus with some overlapping words
corpus_list = [
    "The cat sat on the mat and looked around.",
    "A dog ran across the park chasing a ball.",
    "The cat and the dog played together in the garden.",
    "Running fast, the athlete won the gold medal.",
    "The bird flew over the garden where the cat and dog played."
]

# Function to generate N-grams
def generate_ngrams(text, n):
    tokens = text.lower().split()
    return list(ngrams(tokens, n))

# Processing all texts and collecting n-gram counts
bigram_counts = Counter()
trigram_counts = Counter()
for corpus in corpus_list:
    bigram_counts.update(generate_ngrams(corpus, 2))
    trigram_counts.update(generate_ngrams(corpus, 3))

# Function for Laplace Smoothing
def laplace_smoothing(ngram_counts, vocab_size, alpha=1):
    total_ngrams = sum(ngram_counts.values())
    return {ngram: (count + alpha) / (total_ngrams + alpha * vocab_size) for ngram, count in ngram_counts.items()}

# Applying Laplace smoothing
vocab_size = len(set(" ".join(corpus_list).lower().split()))  # Unique words in corpus
laplace_bigram = laplace_smoothing(bigram_counts, vocab_size)

# -------------------
# Attention Mechanism
# -------------------

class SimpleAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super(SimpleAttention, self).__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, embeddings):
        q = self.query(embeddings)
        k = self.key(embeddings)
        v = self.value(embeddings)

        attention_scores = torch.matmul(q, k.T) / np.sqrt(k.shape[-1])
        attention_weights = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, v)

# Fake word embeddings (just for demo, normally from pre-trained models)
word_embeddings = {word: torch.randn(10) for word in set(" ".join(corpus_list).lower().split())}

# Adding a fallback for missing words
def get_word_embedding(word, word_embeddings):
    # If the word is missing in embeddings, create a default random tensor for it
    return word_embeddings.get(word, torch.randn(10))

# ------------------------
# Calculate Model Probabilities for different bigrams
# ------------------------

# Function to get the probability of a bigram based on raw counts and Laplace smoothed values
def get_bigram_probability(bigram, bigram_counts, laplace_bigram):
    raw_prob = bigram_counts.get(bigram, 0) / sum(bigram_counts.values())
    laplace_prob = laplace_bigram.get(bigram, 0)
    return raw_prob, laplace_prob

# List of bigrams to compare
bigrams_to_compare = [("the", "cat"), ("cat", "sat"), ("dog", "ran"), ("the", "dog"), ("gold", "medal")]

# Calculate probabilities for each bigram and model
raw_probs = []
laplace_probs = []
attention_probs = []

for bigram in bigrams_to_compare:
    raw_prob, laplace_prob = get_bigram_probability(bigram, bigram_counts, laplace_bigram)
    raw_probs.append(raw_prob)
    laplace_probs.append(laplace_prob)
    # For attention, we use the attention context representation
    context_vectors = torch.stack([get_word_embedding(word, word_embeddings) for word in bigram])
    attention_model = SimpleAttention(embed_dim=10)
    attention_rep = attention_model(context_vectors)
    attention_prob = attention_rep.mean().item()  # Using the mean of the attention vector as a proxy for probability
    attention_probs.append(attention_prob)

# ---------------------------
# Plotting Individual Model Probabilities
# ---------------------------

# Plot Raw Bigram Probabilities
plt.figure(figsize=(12, 8))
plt.bar([str(bigram) for bigram in bigrams_to_compare], raw_probs, color='blue')
plt.title('Raw Bigram Probabilities')
plt.xlabel('Bigram')
plt.ylabel('Probability')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('raw_bigram_probabilities.png')
plt.show()

# Plot Laplace Smoothed Probabilities
plt.figure(figsize=(12, 8))
plt.bar([str(bigram) for bigram in bigrams_to_compare], laplace_probs, color='orange')
plt.title('Laplace Smoothed Probabilities')
plt.xlabel('Bigram')
plt.ylabel('Probability')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('laplace_smoothed_probabilities.png')
plt.show()

# Plot Attention-based Probabilities
plt.figure(figsize=(12, 8))
plt.bar([str(bigram) for bigram in bigrams_to_compare], attention_probs, color='green')
plt.title('Attention-based Probabilities')
plt.xlabel('Bigram')
plt.ylabel('Probability')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('attention_based_probabilities.png')
plt.show()

# ---------------------------
# Combined Comparison Graph
# ---------------------------

# Combined Comparison of All Models
plt.figure(figsize=(12, 8))
width = 0.2
x = np.arange(len(bigrams_to_compare))

plt.bar(x - width, raw_probs, width, label='Raw Bigram', color='blue')
plt.bar(x, laplace_probs, width, label='Laplace Smoothing', color='orange')
plt.bar(x + width, attention_probs, width, label='Attention-based', color='green')

plt.title('Comparison of Probabilities: Raw, Laplace, Attention')
plt.xlabel('Bigram')
plt.ylabel('Probability')
plt.xticks(x, [str(bigram) for bigram in bigrams_to_compare], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('combined_comparison.png')
plt.show()
