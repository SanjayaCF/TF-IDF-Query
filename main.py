import os
import re
import math
import pickle
from collections import Counter, defaultdict
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns


# Stemmer Bahasa Indonesia
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()


#Fungsi untuk preprocessing
def preprocess_text(text):
    # normalisasi dengan mengubah semua huruf ke huruf kecil
    text = text.lower()
    # menghilangkan semua tanda baca dan alphabet dengan regex
    text = re.sub(r'[^a-z\s]', ' ', text)
    # tokenisasi
    tokens = word_tokenize(text)
    # menghilangkan stopword untuk bahasa indonesia
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    # stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens


#Fungsi untuk membaca dan memproses dokumen
def load_document(folder_path):
    documents = {}
    doc_term_freq = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = preprocess_text(text)
            documents[filename] = tokens
            doc_term_freq[filename] = Counter(tokens)
    return documents, doc_term_freq

#Fungsi untuk menghitung DF berdasarkan rumus
def hitung_df(documents):
    df = defaultdict(int)
    for tokens in documents.values():
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] += 1
    return df

#Fungsi untuk menghitung TF-IDF berdasarkan rumus
def hitung_tf_idf(doc_term_freq, N, df):
    doc_tf_idf = {}
    for doc, term_counts in doc_term_freq.items():
        tf_idf = {}
        for term, count in term_counts.items():
            tf = count 
            idf = math.log(N / df[term]) if df[term] else float(0)
            tf_idf[term] = tf * idf
        doc_tf_idf[doc] = tf_idf
    return doc_tf_idf

# Fungsi untuk Meenghitugn cosing similarity berdasarkan rumus
def cosine_similarity(vec1, vec2):
    dot = sum(vec1.get(term, 0.0) * vec2.get(term, 0.0) for term in set(vec1.keys()).union(set(vec2.keys())))
    norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val**2 for val in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


folder_tujuan = r"artikel" # menggunakan raw string, dalam kasus jika ingin menggunakan absolute path
doc_pickle_file = "documents.pkl"
index_pickle_file = "index.pkl"

# variable global
documents = {}
doc_term_freq = {}
doc_tf_idf = {}

# cek apakah file hasil pemrosesan dokumen sudah ada
if os.path.exists(doc_pickle_file):
    with open(doc_pickle_file, "rb") as f:
        documents, doc_term_freq = pickle.load(f)
    print("Dokumen telah dimuat dari file pickle.")
else:
    print("Memproses dokumen....")
    documents, doc_term_freq = load_document(folder_tujuan)
    with open(doc_pickle_file, "wb") as f:
        pickle.dump((documents, doc_term_freq), f)
    print("Dokumen telah disimpan ke file pickle.")

# Cek apakah indeks tf-idf sudah ada
if os.path.exists(index_pickle_file):
    user_choice = input("Index pickle sudah ada. Apakah ingin memuat indeks tf-idf yang sudah ada? (y/n): ")
    if user_choice.lower() == 'y':
        with open(index_pickle_file, "rb") as f:
            doc_tf_idf = pickle.load(f)
        print("Indeks tf-idf dimuat dari file pickle.")
    else:
        print("Menghitung ulang indeks tf-idf...")
        N = len(documents)
        df = hitung_df(documents)
        doc_tf_idf = hitung_tf_idf(doc_term_freq, N, df)
        with open(index_pickle_file, "wb") as f:
            pickle.dump(doc_tf_idf, f)
        print("Indeks tf-idf telah dihitung dan disimpan ke file pickle.")
else:
    print("Index pickle tidak ditemukan. Menghitun indeks tf-idf...")
    N = len(documents)
    df = hitung_df(documents)
    doc_tf_idf = hitung_tf_idf(doc_term_freq, N, df)
    with open(index_pickle_file, "wb") as f:
        pickle.dump(doc_tf_idf, f)
    print("Indeks tf-idf telah dihitung dan disimpan ke file pickle.")



# Jika ingin menyimpan index dalam csv
# with open("index.csv", "w", newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     # Menulis header
#     writer.writerow(["Dokumen", "Term", "TF-IDF"])
#     # Menulis setiap entri indeks
#     for doc, tfidf in doc_tf_idf.items():
#         for term, value in tfidf.items():
#             writer.writerow([doc, term, value])



# Frekuensi token dari seluruh dokumen untuk visualisasi
all_tokens = []
for tokens in documents.values():
    all_tokens.extend(tokens)
all_tokens_freq = Counter(all_tokens)

# WordCloud dari token semua dokumen
wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(all_tokens_freq)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud Seluruh Dokumen")
plt.show()

# Top 10 token dari semua dokumen
top10_all = all_tokens_freq.most_common(10)
terms, freqs = zip(*top10_all)
plt.figure(figsize=(8, 5))
plt.bar(terms, freqs, color='skyblue')
plt.xlabel("Token")
plt.ylabel("Frekuensi")
plt.title("Top 10 Token Seluruh Dokumen")
plt.show()

# Heatmap token dari semua dokumen
all_terms = set()
for tfidf in doc_tf_idf.values():
    all_terms.update(tfidf.keys())
all_terms = list(all_terms)

df_tfidf = pd.DataFrame(0.0, index=documents.keys(), columns=all_terms)
for doc, tfidf in doc_tf_idf.items():
    for term, value in tfidf.items():
        df_tfidf.loc[doc, term] = value

plt.figure(figsize=(12, 8))
sns.heatmap(df_tfidf, cmap='viridis')
plt.title("Heatmap TF-IDF per Dokumen dan Term")
plt.xlabel("Term")
plt.ylabel("Dokumen")
plt.show()

# Query
query = input("Masukkan query: ")
query_tokens = preprocess_text(query)
query_term_freq = Counter(query_tokens)

# Menghitung DF
df = hitung_df(documents)

# Menghitung TF-IDF berdasarkan query
query_tf_idf = {}
for term, count in query_term_freq.items():
    if term in df:
        idf = math.log(len(documents) / df[term])
        query_tf_idf[term] = count * idf
    else:
        query_tf_idf[term] = 0.0

# Menghitung cosine similarity nya
doc_scores = {}
for doc, tfidf_vector in doc_tf_idf.items():
    score = cosine_similarity(tfidf_vector, query_tf_idf)
    doc_scores[doc] = score

# Mengurutkan dokumen berdasarkan cosine similarity
sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

print("\nHasil pencarian berdasarkan cosine similarity:")
for doc, score in sorted_docs:
    print(f"Dokumen: {doc}, Cosine Similarity: {score:.4f}")

# Visualisasi hasil pencarian berdasarkan cosine similarity
docs = list(doc_scores.keys())
scores = [doc_scores[doc] for doc in docs]

plt.figure(figsize=(10, 5))
plt.bar(docs, scores, color='orange')
plt.xlabel("Dokumen")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity antara Query dan Dokumen")
plt.xticks(rotation=45)
plt.show()
