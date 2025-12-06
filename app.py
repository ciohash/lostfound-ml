# app_streamlit_model_pretty.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json
import os
from math import isnan

# ---------- CONFIG ----------
DATA_CSV = "lost_found_dataset_realistic_metadata_30k.csv"
TFIDF_PKL = "tfidf.joblib"
MATRIX_NPZ = "precomputed_matrices.npz"
SCALER_PKL = "scaler.joblib"
MODEL_PKL = "model_lr.joblib"      # or model_rf.joblib
FEEDBACK_JSON = "feedback.json"
TOPK_PREFILTER = 200               # fast prefilter by TF-IDF (tuneable)
RE_RANK_TOPK = 50                  # show top-50 after model scoring (UI limit)
# -----------------------------

st.set_page_config(
    page_title="Campus Lost & Found — AutoMatch",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Utils ----------
@st.cache_data
def load_resources():
    df = pd.read_csv(DATA_CSV).reset_index(drop=True)
    # parse timestamp if exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT
    if 'text_blob' not in df.columns:
        df['text_blob'] = (df['item_name'].fillna('') + ' ' +
                           df['description'].fillna('') + ' ' +
                           df['color'].fillna('') + ' ' +
                           df['brand'].fillna('') + ' ' +
                           df['location'].fillna(''))
    tf = joblib.load(TFIDF_PKL)
    tfidf_matrix = sparse.load_npz(MATRIX_NPZ)
    scaler = joblib.load(SCALER_PKL)
    model = joblib.load(MODEL_PKL)
    return df, tf, tfidf_matrix, scaler, model

def jaccard_tokens(a, b):
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb)) / len(sa | sb)

def time_diff_hours(t1, t2):
    try:
        if pd.isna(t1) or pd.isna(t2):
            return 99999.0
        return abs((t1 - t2).total_seconds()) / 3600.0
    except Exception:
        return 99999.0

def confidence_badge(score):
    # score is probability 0..1
    if score >= 0.85:
        return "High", "✅ High confidence"
    elif score >= 0.6:
        return "Medium", "⚠️ Medium confidence"
    else:
        return "Low", "❌ Low confidence"

def save_feedback(lost_item_id, found_item_id, label):
    rec = {"lost_id": int(lost_item_id), "found_id": int(found_item_id), "label": int(label), "ts": datetime.now().isoformat()}
    if os.path.exists(FEEDBACK_JSON):
        try:
            with open(FEEDBACK_JSON, "r") as f:
                fb = json.load(f)
        except Exception:
            fb = []
    else:
        fb = []
    fb.append(rec)
    with open(FEEDBACK_JSON, "w") as f:
        json.dump(fb, f, indent=2)
    return True

# ---------- Load ----------
with st.spinner("Loading model and dataset..."):
    df, tf, tfidf_matrix, scaler, model = load_resources()

# Sidebar - stats and controls
st.sidebar.header("Dataset & Model")
st.sidebar.write("Rows:", df.shape[0])
st.sidebar.write("Lost items:", int((df['lost_or_found']=='lost').sum()))
st.sidebar.write("Found items:", int((df['lost_or_found']=='found').sum()))
st.sidebar.markdown("---")
st.sidebar.write("Model:", os.path.basename(MODEL_PKL))
st.sidebar.write("TF-IDF features:", getattr(tf, "vocabulary_", None) and len(tf.vocabulary_) or "unknown")
st.sidebar.markdown("---")
st.sidebar.caption("Suggestions: Describe the object clearly (color, brand, unique mark).")

# Top bar
st.markdown("<h1 style='text-align:center'>Campus Lost & Found — <span style='color:#ff4b4b'>AutoMatch</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>Write a description below — the system finds the best candidate found items and shows confidence & metadata.</p>", unsafe_allow_html=True)
st.markdown("---")

# Two-column layout: left = input, right = recent feedback & quick actions
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Describe the lost item")
    item_name_input = st.text_input("Item name (optional)", "")
    description_input = st.text_area("Describe what you lost (include color, brand, unique marks).", height=140)
    color_input = st.text_input("Color (optional)", "")
    location_hint = st.text_input("Last seen location (optional)", "")
    find_k = st.slider("Number of results to show", 1, 50, 8)

    if st.button("Find Matches", type="primary"):
        if not description_input.strip():
            st.error("Please enter a description so the system can search.")
        else:
            # build lost blob
            lost_blob = (item_name_input + " " + description_input + " " + color_input + " " + location_hint).strip()
            lost_vec = tf.transform([lost_blob])

            # prefilter: top-N by TF-IDF cosine from all found items
            found_indices = df[df['lost_or_found']=='found'].index.tolist()
            cos_all = cosine_similarity(lost_vec, tfidf_matrix[found_indices]).flatten()

            # get top TOPK_PREFILTER positions
            if len(cos_all) <= TOPK_PREFILTER:
                top_pos = np.argsort(-cos_all)
            else:
                top_pos = np.argpartition(-cos_all, TOPK_PREFILTER-1)[:TOPK_PREFILTER]
                top_pos = top_pos[np.argsort(-cos_all[top_pos])]

            candidate_found_indices = [found_indices[i] for i in top_pos]
            cos_subset = cos_all[top_pos]
            found_rows = df.loc[candidate_found_indices].reset_index(drop=True)

            # build features
            feats = []
            for i, fi in enumerate(candidate_found_indices):
                found = found_rows.iloc[i]
                feat = {
                    'cosine_text': float(cos_subset[i]),
                    'jaccard_desc': jaccard_tokens(description_input, found['description']),
                    'color_match': 1 if color_input.strip().lower() and color_input.strip().lower()==str(found.get('color','')).strip().lower() else 0,
                    'brand_match': 1 if str(item_name_input).strip().lower() and str(item_name_input).strip().lower()==str(found.get('item_name','')).strip().lower() else 0,
                    'location_match': 1 if location_hint.strip().lower() and location_hint.strip().lower()==str(found.get('location','')).strip().lower() else 0,
                    'user_match': 0,
                    'time_diff_hours': 99999.0,
                    'len_diff': abs(len(description_input) - len(str(found['description']))),
                    'name_match': 1 if item_name_input.strip().lower() and item_name_input.strip().lower()==str(found.get('item_name','')).strip().lower() else 0
                }
                feats.append(feat)
            feats_df = pd.DataFrame(feats)
            X = feats_df[[
                'cosine_text','jaccard_desc','color_match','brand_match',
                'location_match','user_match','time_diff_hours','len_diff','name_match'
            ]].fillna(0.0).astype(float)
            Xs = scaler.transform(X)
            probs = model.predict_proba(Xs)[:,1]

            found_rows = found_rows.copy()
            found_rows['score'] = probs
            results = found_rows.sort_values('score', ascending=False).head(find_k)

            # show results as cards
            st.success(f"Found {len(results)} matches (re-ranked from top {TOPK_PREFILTER} candidates).")
            for idx, row in results.reset_index(drop=True).iterrows():
                score = float(row['score'])
                lvl, badge = confidence_badge(score)
                # Card-like display
                with st.container():
                    rcol1, rcol2 = st.columns([4,1])
                    with rcol1:
                        # Item name as main heading
                        st.markdown(f"### {row['item_name']}")
                        
                        # Description
                        if pd.notna(row.get('description', '')) and str(row['description']).strip():
                            st.write(f"_{row['description']}_")
                        
                        # Key details in a compact format
                        details = []
                        if row.get('color', ''):
                            details.append(f"🎨 {row['color']}")
                        if row.get('brand', ''):
                            details.append(f"🏷️ {row['brand']}")
                        if row.get('location', ''):
                            details.append(f"📍 {row['location']}")
                        
                        if details:
                            st.markdown(" | ".join(details))
                        
                        # Additional info in expander
                        with st.expander("More details"):
                            ts = row.get('timestamp', None)
                            if pd.notna(ts):
                                st.write(f"**Found time:** {pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')}")
                            if row.get('user', ''):
                                st.write(f"**Reported by:** {row.get('user','')}")
                            if row.get('brand', ''):
                                st.write(f"**Brand:** {row.get('brand','')}")
                            if row.get('color', ''):
                                st.write(f"**Color:** {row.get('color','')}")
                            if row.get('location', ''):
                                st.write(f"**Location:** {row.get('location','')}")
                    
                    with rcol2:
                        # Score display
                        st.markdown(f"<div style='text-align:center;padding:12px;border-radius:8px;background:#f5f5f5;margin-bottom:8px'>"
                                    f"<div style='font-size:24px;font-weight:700;color:#1f77b4'>{score:.3f}</div>"
                                    f"<div style='color:gray;font-size:12px;margin-top:4px'>{badge}</div></div>", unsafe_allow_html=True)
                        
                        # Action buttons
                        if st.button("✓ Correct", key=f"c_{int(row['item_id'])}", use_container_width=True):
                            save_feedback(chosen_item_id := "user_input", int(row['item_id']), 1)
                            st.success("Marked as correct — thanks!")
                        if st.button("✗ Incorrect", key=f"i_{int(row['item_id'])}", use_container_width=True):
                            save_feedback(chosen_item_id := "user_input", int(row['item_id']), 0)
                            st.info("Marked as incorrect — thanks!")
                st.markdown("---")

with col2:
    st.subheader("Quick actions & diagnostics")
    if st.button("Show sample dataset rows"):
        st.write(df.sample(8).reset_index(drop=True)[['item_id','lost_or_found','item_name','description','color','brand','location','user','timestamp']])
    st.markdown("### Quick stats")
    st.metric("Total rows", df.shape[0])
    st.metric("Lost items", int((df['lost_or_found']=='lost').sum()))
    st.metric("Found items", int((df['lost_or_found']=='found').sum()))
    st.markdown("---")
    # evaluation helper
    if st.button("Compute fast eval metrics (Top-100 prefilter)"):
        st.info("Running evaluation — this may take a few minutes depending on dataset size.")
        # call lightweight internal evaluation over eval file if exists
        EVAL_CSV = "lost_found_exact_match_pairs.csv"
        if not os.path.exists(EVAL_CSV):
            st.error("Exact-match eval file not found in project root.")
        else:
            eval_pairs = pd.read_csv(EVAL_CSV)
            # quickly compute metrics using the same prefilter logic but only on eval pairs
            found_indices_all = df[df['lost_or_found']=='found'].index.tolist()
            top1 = top3 = top5 = 0
            mrr = 0.0
            total = 0
            # We'll implement a simple loop with lightweight progress below
            total_pairs = len(eval_pairs)
            pbar = st.progress(0)
            processed = 0
            for _, er in eval_pairs.iterrows():
                lid = int(er['lost_id'])
                fid = int(er['found_id'])
                # map to df indexes
                lost_idx_list = df.index[df['item_id']==lid].tolist()
                found_idx_list = df.index[df['item_id']==fid].tolist()
                if not lost_idx_list or not found_idx_list:
                    continue
                lost_idx = lost_idx_list[0]
                true_found_idx = found_idx_list[0]

                lost_vec = tf.transform([df.loc[lost_idx,'text_blob']])
                cos_all = cosine_similarity(lost_vec, tfidf_matrix[found_indices_all]).flatten()
                if len(cos_all) <= TOPK_PREFILTER:
                    top_pos = np.argsort(-cos_all)
                else:
                    top_pos = np.argpartition(-cos_all, TOPK_PREFILTER-1)[:TOPK_PREFILTER]
                    top_pos = top_pos[np.argsort(-cos_all[top_pos])]
                candidate_found_indices = [found_indices_all[i] for i in top_pos]
                cos_subset = cos_all[top_pos]
                # build features for candidates
                feats_df = []
                for i, fi in enumerate(candidate_found_indices):
                    found = df.loc[fi]
                    feats_df.append({
                        'cosine_text': float(cos_subset[i]),
                        'jaccard_desc': jaccard_tokens(df.loc[lost_idx,'description'], found['description']),
                        'color_match': 1 if str(df.loc[lost_idx,'color']).strip().lower()==str(found.get('color','')).strip().lower() else 0,
                        'brand_match': 1 if str(df.loc[lost_idx,'brand']).strip().lower()==str(found.get('brand','')).strip().lower() else 0,
                        'location_match': 1 if str(df.loc[lost_idx,'location']).strip().lower()==str(found.get('location','')).strip().lower() else 0,
                        'user_match': 1 if str(df.loc[lost_idx,'user']).strip().lower()==str(found.get('user','')).strip().lower() else 0,
                        'time_diff_hours': time_diff_hours(df.loc[lost_idx,'timestamp'], found.get('timestamp', None)),
                        'len_diff': abs(len(str(df.loc[lost_idx,'description'])) - len(str(found['description']))),
                        'name_match': 1 if str(df.loc[lost_idx,'item_name']).strip().lower()==str(found.get('item_name','')).strip().lower() else 0
                    })
                feats_df = pd.DataFrame(feats_df)
                X = feats_df[[
                    'cosine_text','jaccard_desc','color_match','brand_match',
                    'location_match','user_match','time_diff_hours','len_diff','name_match'
                ]].fillna(0.0).astype(float)
                Xs = scaler.transform(X)
                probs = model.predict_proba(Xs)[:,1]
                ranked_local = np.argsort(-probs)
                ranked_found_indices = [candidate_found_indices[i] for i in ranked_local]
                # metrics
                if true_found_idx in ranked_found_indices[:1]:
                    top1 += 1
                if true_found_idx in ranked_found_indices[:3]:
                    top3 += 1
                if true_found_idx in ranked_found_indices[:5]:
                    top5 += 1
                # mrr
                pos = None
                for pos_i, fidx in enumerate(ranked_found_indices, start=1):
                    if fidx == true_found_idx:
                        pos = pos_i
                        break
                if pos:
                    mrr += 1.0/pos
                processed += 1
                if processed % 50 == 0:
                    pbar.progress(min(processed/total_pairs, 1.0))
            if processed > 0:
                st.success("Evaluation completed.")
                st.write("Pairs processed:", processed)
                st.write("Top-1:", top1/processed)
                st.write("Top-3:", top3/processed)
                st.write("Top-5:", top5/processed)
                st.write("MRR:", mrr/processed)
            else:
                st.warning("No eval pairs processed (check item_id mapping).")
    st.markdown("---")
    st.markdown("Feedback saved to `feedback.json`. Click items' Mark buttons to append feedback.")
    st.caption("If you want prettier UI elements (icons, custom CSS), we can add them next.")
