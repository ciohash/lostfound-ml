🎒 Campus Lost & Found — AutoMatch (Classical ML System)
A high-accuracy Lost–Found item matcher built using ONLY classical machine learning — no deep learning.
<p align="center"> <img src="https://img.shields.io/badge/ML-Classical%20ML-blue?style=for-the-badge"/> <img src="https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge"/> <img src="https://img.shields.io/badge/Accuracy-95%25-green?style=for-the-badge"/> <img src="https://img.shields.io/badge/Ranking-MRR%200.97-yellow?style=for-the-badge"/> </p>
🚀 Overview

This project builds a complete Lost & Found item matching system using:

TF-IDF text embeddings

Metadata similarity features

Classical ML models (Logistic Regression / RandomForest)

Top-K retrieval & reranking

Streamlit UI for interactive search

✅ Strictly no deep learning — fully compliant with project requirements.

🌟 Highlights

🔍 Automatic matching of lost–found items

✍️ Handles typos, slang, incomplete descriptions

🧠 Classical ML: TF-IDF + Logistic Regression / RandomForest

📍 Metadata-aware: location, color, brand, user, timestamp

⚡ Fast retrieval using Top-100 cosine prefiltering

📈 High ranking performance

🌐 Easy deployment with Streamlit Cloud (Free)

📊 Performance
Ranking Metrics (Top-K Retrieval)

Evaluated on 12,017 lost–found pairs:

Metric	Score
Top-1 Accuracy	0.9502
Top-3 Accuracy	0.9998
Top-5 Accuracy	1.0000
MRR	0.9749
Classification Metrics
Metric	Score
Best Threshold	0.9800
F1 Score	0.8850
Precision	0.7937
Recall	1.0000
AUC	0.9989
📘 Dataset

A realistic 30,000-item dataset was created, containing:

✔ Text Descriptions

Natural sentences

Typos (“bottel”, “hedfone”)

Slang (“yaar”, “bro”, “idk”)

Incomplete descriptions

Mixed language styles

✔ Metadata

color

brand

location

user

timestamp

category

item_name

✔ Noise

20% found items intentionally mismatched

Wrong locations

Incorrect color or brand

✔ Ground-Truth File

lost_found_exact_match_pairs.csv for Top-K evaluation.

🧠 Machine Learning Pipeline
1️⃣ TF-IDF Vectorization

30k vocabulary

Uni/Bi-grams

Sparse .npz matrix

2️⃣ Feature Engineering

For each (lost, found) pair:

cosine_text

jaccard_desc

color_match

brand_match

location_match

user_match

time_diff_hours

len_diff

name_match

3️⃣ Supervised Classification

Models used:

Logistic Regression

RandomForest

Both achieved AUC ≈ 1.0 on the pairwise task.

4️⃣ Ranking System

Candidate retrieval (Top-100 cosine similarity)

Feature computation

ML scoring

Sorted Top-K matches

🧪 Sample Lost Item Inputs (Test Cases)

You can paste these into your Streamlit UI:

🔹 Test Case 1 — Lost Keychain
Description: Keychain (black) lost near walkway.
Category: Bag
Location lost: walkway
Time lost: 2025-02-18 14:30
Brand: Samsung 

🔹 Test Case 2 — Lost Water Bottle
Description: Blue metal bottle with a dent at the bottom.
Category: Bottle
Location lost: Canteen
Time lost: 2025-02-17 11:00

🔹 Test Case 3 — Lost Notebook
Description: Red spiral notebook with ML class notes.
Category: Notebook
Location lost: Block B classroom
Time lost: 2025-02-19 09:15

🔹 Test Case 4 — Lost Umbrella
Description: Small foldable dark blue umbrella with silver handle.
Category: Umbrella
Location lost: Parking Area
Time lost: 2025-02-20 18:45

🔹 Test Case 5 — Lost Watch
Description: Watch (black) lost near walkway.
Category: Electronics
Location lost: walkway
Time lost: 2025-02-16 20:10
Brand: Casio 

🛠️ How to Run the Project
Install Requirements
pip install -r requirements.txt

Prepare TF-IDF + Training Data
python prepare_pairs.py

Train ML Models
python train_model.py

Evaluate Retrieval Performance
python evaluate_retrieval_fast.py

Launch Streamlit App
streamlit run app.py

🌐 Deploy to Streamlit Cloud (Free)

Deployment steps (5 minutes):

Push repo to GitHub

Visit https://share.streamlit.io

Connect your repository

Choose app.py

Deploy ✨

Your app gets a free public URL:
👉 https://your-app-name.streamlit.app

📌 Future Enhancements

Hard-negative sampling

Color normalization (navy → blue)

Synonym expansion (bottle/flask/tumbler)

ANN search (FAISS / Annoy)

User-feedback-based retraining

Optional image metadata integration

✔️ Conclusion

This project delivers a fully functional Campus Lost & Found AutoMatch System using classical machine learning. It achieves extremely high accuracy, handles real-world noisy text, includes metadata reasoning, and provides an easy-to-use Streamlit interface.

It meets every ML project requirement:

Classical ML only

Dataset created

Feature engineering

Ranking system

Evaluation metrics

Streamlit demonstration

Clean implementation
