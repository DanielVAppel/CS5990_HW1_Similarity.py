# -------------------------------------------------------------------------
# AUTHOR: Daniel Appel
# FILENAME: Similarity.py
# SPECIFICATION: Finds and outputs the two most similar documents from the cleaned_documents.csv dataset based on their cosine similarity using only fundamental Python functionalities and avoiding advanced libraries. 
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 3 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Import required library
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

# Read the documents from the CSV file
with open('cleaned_documents.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skip header
            # Ensure each document is stored as a single string
            documents.append(' '.join(row).split())  # Tokenize words properly

# Extract unique words to build the vocabulary
vocabulary = []
for doc in documents:
    for word in doc:
        if word not in vocabulary:
            vocabulary.append(word)

# Build document-term matrix (binary encoding)
doc_term_matrix = []
for doc in documents:
    row_vector = [1 if term in doc else 0 for term in vocabulary]
    doc_term_matrix.append(row_vector)

# Compute cosine similarity for each document pair
max_similarity = -1  # Initialize to ensure an update
most_similar_docs = (-1, -1)

for i in range(len(doc_term_matrix)):
    for j in range(i + 1, len(doc_term_matrix)):
        similarity = cosine_similarity([doc_term_matrix[i]], [doc_term_matrix[j]])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_docs = (i + 1, j + 1)  # Adjust for 1-based indexing

# Print the highest cosine similarity in the required format
print(f"The most similar documents are document {most_similar_docs[0]} and document {most_similar_docs[1]} with cosine similarity = {max_similarity:.4f}")