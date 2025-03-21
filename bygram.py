import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter
from nltk.util import ngrams
import nltk

# Custom text for testing
custom_text = """
The quick brown fox jumps over the lazy dog. The dog barked loudly at the fox.
The fox ran away quickly. A cat watched the scene from a nearby tree.
The dog chased the fox but stopped after a while. The quick brown fox was too fast.
The cat climbed down the tree and walked away silently. The dog returned to its spot.
"""

# Tokenize the custom text into words
text = custom_text.split()

# Generate bigrams from the custom text
bigrams = list(ngrams(text, 2))
bigram_freq = Counter(bigrams)

# Top 10 most common bigrams
top_10_bigrams = bigram_freq.most_common(10)

# Plotting
labels, values = zip(*top_10_bigrams)
labels = [' '.join(bigram) for bigram in labels]  # Convert bigram tuples to strings
plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Most Frequent Bigrams')
plt.xlabel('Bigrams')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout to prevent label overlap

# Save the plot to a file
plt.savefig('custom_bigrams_plot.png')
print("Plot saved as 'custom_bigrams_plot.png'")
