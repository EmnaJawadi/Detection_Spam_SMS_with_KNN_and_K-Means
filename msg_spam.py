import streamlit as st
import string
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Configuration de la page
st.set_page_config(page_title="Détection Spam SMS", layout="wide")
st.title(" Détection de Spam SMS avec KNN et K-Means")

# Téléchargement des stopwords
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


# Définir la fonction à l'extérieur pour qu'elle soit "picklable"
def process_text(text):
    """Nettoie le texte : supprime ponctuation et mots inutiles"""
    nopunc = ''.join([c for c in text if c not in string.punctuation])
    clean_words = [w for w in nopunc.split() if w.lower() not in stopwords.words('english')]
    return ' '.join(clean_words)


# =============================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# =============================================================================
@st.cache_data
def load_and_process_data():
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

    df['clean_message'] = df['message'].apply(process_text)

    tfidf_vectorizer = TfidfVectorizer(max_features=2500)
    X_tfidf = tfidf_vectorizer.fit_transform(df['clean_message'])

    return df, X_tfidf, df['label'], tfidf_vectorizer


with st.spinner('Chargement et prétraitement des données...'):
    df, X_tfidf, y, vectorizer = load_and_process_data()

# =============================================================================
# ÉTAPE 2 : EXPLORATION
# =============================================================================
st.header("1. Exploration des données")
col1, col2 = st.columns(2)
with col1:
    st.write("Aperçu du dataset :")
    st.dataframe(df[['label', 'message']].head())
with col2:
    st.write("Répartition des classes :")
    fig, ax = plt.subplots(figsize=(5, 3))
    df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)
    st.pyplot(fig)

# =============================================================================
# ÉTAPE 3 : K-MEANS 
# =============================================================================
st.header("2. Analyse Non Supervisée (K-Means)")


# AJOUT DU "_" DEVANT X POUR EVITER L'ERREUR DE HASHAGE
@st.cache_resource
def run_kmeans(_X):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(_X)
    return kmeans


kmeans = run_kmeans(X_tfidf)
df['kmeans_cluster'] = kmeans.labels_

contingency_table = pd.crosstab(df['label'], df['kmeans_cluster'])
st.write("### Tableau de Contingence")
st.dataframe(contingency_table)

# =============================================================================
# ÉTAPE 4 : KNN
# =============================================================================
st.header("3. Classification Supervisée (KNN)")

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.25, random_state=42)


@st.cache_resource
def train_knn_grid_search(_X_train, y_train):
    param_grid = {'n_neighbors': list(range(1, 21))}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(_X_train, y_train)
    return grid_search


with st.spinner("Entraînement du modèle KNN..."):
    grid_search = train_knn_grid_search(X_train, y_train)
    best_knn = grid_search.best_estimator_
    y_pred_best = best_knn.predict(X_test)

col_res1, col_res2 = st.columns(2)
with col_res1:
    st.success(f"Meilleur k trouvé : {grid_search.best_params_['n_neighbors']}")
    st.info(f"Précision : {accuracy_score(y_test, y_pred_best):.4f}")
    st.text("Rapport de classification :")
    st.text(classification_report(y_test, y_pred_best))

with col_res2:
    st.subheader("Performance selon K")
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    fig_cv, ax_cv = plt.subplots(figsize=(6, 4))
    plt.plot(range(1, 21), mean_test_scores, marker='o', linestyle='-')
    plt.grid(True)
    st.pyplot(fig_cv)

# =============================================================================
# ÉTAPE 6 : TEST
# =============================================================================
st.markdown("---")
st.header(" Testez le modèle")
user_input = st.text_area("Votre message :", "WINNER!! You have won a prize! Txt CLAIM to 12345")

if st.button("Analyser le message"):
    if user_input:
        clean_msg = process_text(user_input)
        vectorized_msg = vectorizer.transform([clean_msg])
        prediction = best_knn.predict(vectorized_msg)[0]

        if prediction == 'spam':
            st.error(" SPAM DETECTÉ")
        else:
            st.success(" MESSAGE NORMAL (HAM)")
