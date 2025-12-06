# lostfound-ml
A complete machine-learning–based Lost &amp; Found item matching system built using ONLY classical ML techniques — no deep learning


🏫 Campus Lost & Found — AutoMatch (Classical ML System)

A complete machine-learning–based Lost & Found item matching system built using ONLY classical ML techniques — no deep learning, as required by the project guidelines .

This system automatically matches lost and found item descriptions using:

TF-IDF text vectors

Engineered metadata features

Logistic Regression / RandomForest pairwise classifier

Ranking system with Top-K evaluation

Streamlit UI for user interaction

We built a realistic dataset with metadata, noise, typos, slang, mismatched items, and exact-match pairs.

📌 Project Features

✔ Automatic match suggestions for lost–found items
✔ Uses classical ML (TF-IDF + logistic regression / random forest)
✔ Supports typos, slang, mismatched items, and noisy descriptions
✔ Includes metadata-based matching (color, brand, location, user, timestamp)
✔ Fast evaluation using Top-100 TF-IDF prefilter + ML ranking
✔ Streamlit front-end for demonstration
✔ High accuracy:

Evaluated on 12017 pairs

Top-1 accuracy: 0.9502

Top-3 accuracy: 0.9998

Top-5 accuracy: 1.0000

MRR: 0.9749

=== Classification Metrics ===

Best Threshold: 0.9800

F1 Score: 0.8850

Precision: 0.7937

Recall: 1.0000

AUC: 0.9989

Top-1 Accuracy: 0.9502

Top-3 Accuracy: 0.9998


📘 Dataset Description

We created a highly realistic dataset with:

✔ 30,000 total rows (15k lost + 15k found)
✔ Realistic descriptions containing:

Natural sentences

Typos

Slang (bro, yaar, lol, idk etc.)

Incomplete descriptions

Multi-style language

✔ Metadata:

item_name

description

color

brand

location

user

timestamp

✔ Noise & Mismatches

~20% found items are intentional mismatches:

wrong item

wrong color

vague description

unclear details

✔ Exact-match evaluation file

lost_found_exact_match_pairs.csv contains ground truth lost–found pairs for computing:

Top-K accuracy

MRR

🧠 ML Approach
1️⃣ TF-IDF Vectorization

Trained on text_blob

30k max features

Uni/bi-grams

Sparse matrix stored as .npz

2️⃣ Feature Engineering

For each (lost, found) pair:

cosine_text (TF-IDF cosine similarity)

jaccard_desc

color_match

brand_match

location_match

user_match

time_diff_hours

len_diff

name_match

3️⃣ Supervised Pairwise Classification

Models:

Logistic Regression (LR)

RandomForest (RF)

Both achieved:

ROC AUC: 1.0 (on pairwise task)

(Expected, because strong features + easy negatives)

4️⃣ Retrieval Ranking System

For each lost item:

Get Top-100 candidates using TF-IDF cosine

Compute engineered features

Apply trained model

Sort by predicted probability

Evaluate using Top-K metrics

📊 Evaluation Results (Fast Ranking)

Using evaluate_retrieval_fast.py, we obtained:

Evaluated on 12017 pairs

Top-1 accuracy: 0.9502

Top-3 accuracy: 0.9998

Top-5 accuracy: 1.0000

MRR: 0.9749

=== Classification Metrics ===

Best Threshold: 0.9800

F1 Score: 0.8850

Precision: 0.7937

Recall: 1.0000

AUC: 0.9989

Top-1 Accuracy: 0.9502

Top-3 Accuracy: 0.9998

These results show excellent real-world matching performance.

🚀 Running the Project (Exact Order)
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Prepare TF-IDF + Pairwise Training Data
python prepare_pairs.py


Outputs:

tfidf.joblib

precomputed_matrices.npz

pairs_train.pkl

3️⃣ Train the ML Models
python train_model.py


Outputs:

model_lr.joblib

model_rf.joblib

scaler.joblib

4️⃣ Fast Evaluation (Top-K Retrieval)
python evaluate_retrieval_fast.py

5️⃣ Run Streamlit App
streamlit run app.py


Opens UI:

Select lost item → get ranked found matches

Free-text search

View scores, metadata

🌐 Deploy to Streamlit Cloud (Free)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on deploying your app to Streamlit Cloud for free.

Quick steps:
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy! Your app will be live at `https://your-app-name.streamlit.app`

🖥️ Streamlit Demo (Features)

Search lost items

Model-based ranking

Confidence scores

Top results with metadata

Slang + typo handling

Timestamp-based scoring


This project covers every requirement  :

✔ Dataset created
✔ ML approach with classical models only
✔ Streamlit-based demonstration
✔ Ranking evaluation (Top-K, MRR)
✔ Explanation of features + results
✔ Source code + README
📌 Potential Future Improvements

Hard negative sampling

Color normalization (navy → blue)

Synonym handling (bottle = flask = tumbler)

Image metadata features (optional per guidelines)

Feedback-based retraining

ANN index (FAISS/Annoy) for ultra-fast retrieval

✔ Conclusion

This project demonstrates a complete Campus Lost & Found AutoMatch system using classical ML techniques that achieves high accuracy, robustness, and real-world usability, closely following the project guidelines.
