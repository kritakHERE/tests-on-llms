import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Simulate some random word embeddings for a vocabulary
word_embeddings = {
    'the': torch.randn(10), 'cat': torch.randn(10), 'sat': torch.randn(10),
    'on': torch.randn(10), 'mat': torch.randn(10), 'dog': torch.randn(10),
    'chased': torch.randn(10), 'a': torch.randn(10), 'ball': torch.randn(10),
    'park': torch.randn(10), 'gold': torch.randn(10), 'medal': torch.randn(10)
}

# Attention mechanism: A simple multi-head attention-like mechanism
class SimpleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SimpleAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # Adding batch dimension
        x = x.unsqueeze(0)
        attn_output, attn_weights = self.attention(x, x, x)
        return attn_output.squeeze(0), attn_weights.squeeze(0)

# Function to predict the next word using attention mechanism
def predict_next_word_attention(context, word_embeddings, embed_dim=10, num_heads=2):
    context = context.lower().split()  # Convert context to lower case and split it
    predictions = []

    # Initialize the attention model
    attention_model = SimpleAttention(embed_dim, num_heads)

    # Convert context words to their embeddings
    context_embeddings = [word_embeddings[word] for word in context if word in word_embeddings]
    if not context_embeddings:
        return predictions  # No valid words in context
    
    # Convert embeddings to tensor for processing
    context_tensor = torch.stack(context_embeddings)

    # Get attention output and weights
    context_rep, attention_weights = attention_model(context_tensor)
    
    # Calculate probabilities for each word in the vocabulary
    word_scores = {}
    for word, embedding in word_embeddings.items():
        word_similarity = torch.cosine_similarity(context_rep.mean(dim=0), embedding, dim=0).item()
        word_scores[word] = word_similarity
    
    # Normalize the scores to get probabilities
    total_score = sum(word_scores.values())
    word_probs = {word: score / total_score for word, score in word_scores.items()}
    
    # Sort the predictions by probability
    sorted_predictions = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top 10 predictions
    top_10_predictions = sorted_predictions[:10]
    predictions = top_10_predictions

    # Display the predictions and their probabilities
    print("Top 10 Predictions and their Probabilities:")
    for word, prob in predictions:
        print(f"Word: {word}, Probability: {prob:.4f}")
    
    return predictions, word_probs

# Visualize the predictions
def plot_predictions(predictions, word_probs):
    words = [word for word, _ in predictions]
    probs = [prob for _, prob in predictions]

    # Plotting the probabilities
    plt.figure(figsize=(10, 6))
    plt.barh(words, probs, color='skyblue')
    plt.xlabel('Probability')
    plt.title('Top 10 Predicted Words with their Probabilities')
    plt.gca().invert_yaxis()  # To display the highest probability at the top
    plt.show()

# Main Execution
context = "the cat sat on the mat"  # Long context for word prediction

# Predict the next word based on the context using attention
predictions, word_probs = predict_next_word_attention(context, word_embeddings)

# Plot the results
plot_predictions(predictions, word_probs)
