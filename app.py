import streamlit as st
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import re

# Custom stop words
custom_stopwords = ["yang", "dan", "di", "dengan", "untuk", "pada", "adalah", "ini", "itu", "atau", "juga"]

# Sidebar with title
st.sidebar.title("UTS Pencarian & Penambangan Web A")

# Data tab
with st.sidebar:
    st.subheader("Deskripsi Data")
    # Data description here
    data = pd.read_csv("DF_PTA.csv")

# Main content in tabs
st.title("UTS Pencarian & Penambangan Web A")
tabs = st.tabs(['Data', 'LDA', 'Modelling', 'Implementasi'])

with tabs[0]:
    st.write(data)

with tabs[1]:
    topik = st.number_input("Masukkan Jumlah Topik yang Diinginkan", 1, step=1, value=5)
    lda_model = None  # Initialize lda_model

    def submit():
        # LDA model training and data processing
        tf = pd.read_csv("df_tf.csv")
        lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
        lda_top = lda.fit_transform(tf)
        # Data with LDA
        nama_clm = [f"Topik {i + 1}" for i in range(topik)]
        U = pd.DataFrame(lda_top, columns=nama_clm)
        data_with_lda = pd.concat([U, data['Label']], axis=1)
        st.write(data_with_lda)

    if st.button("Submit"):
        submit()

with tabs[2]:
    # Model training and selection
    tf = pd.read_csv("df_tf.csv")
    lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
    lda_top = lda.fit_transform(tf)
    nama_clm = [f"Topik {i + 1}" for i in range(topik)]
    U = pd.DataFrame(lda_top, columns=nama_clm)
    data_with_lda = pd.concat([U, data['Label']], axis=1)

    df = data_with_lda.dropna(subset=['Label', 'Label'])

    X = df.drop(columns=['Label']).values
    y = df['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model1 = KNeighborsClassifier(5)
    model1.fit(X_train, y_train)

    model2 = MultinomialNB()
    model2.fit(X_train, y_train)

    model3 = DecisionTreeClassifier()
    model3.fit(X_train, y_train)

    st.write("Pilih metode yang ingin anda gunakan :")
    met1 = st.checkbox("KNN")
    met2 = st.checkbox("Naive Bayes")
    met3 = st.checkbox("Decision Tree")
    if st.button("Pilih"):
        if met1:
            # Model 1 - KNN
            y_pred = model1.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Akurasi KNN: {:.2f}%".format(accuracy * 100))
        elif met2:
            # Model 2 - Naive Bayes
            y_pred = model2.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Akurasi Naive Bayes: {:.2f}%".format(accuracy * 100))
        elif met3:
            # Model 3 - Decision Tree
            y_pred = model3.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Akurasi Decision Tree: {:.2f}%".format(accuracy * 100))
        else:
            st.write("Anda Belum Memilih Metode")

with tabs[3]:
    # Text analysis and implementation
    data = pd.read_csv("DF_PTA.csv")
    data['Abstrak'].fillna("", inplace=True)
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2)

    def preprocess_text(text):
        # Preprocessing steps
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in custom_stopwords]
        cleaned_text = ' '.join(words)
        return cleaned_text

    st.write("Masukkan Abstrak yang Ingin Dianalisis:")
    user_abstract = st.text_area("Abstrak", "")

    if user_abstract:
        preprocessed_abstract = preprocess_text(user_abstract)
        count_vectorizer.fit(data['Abstrak'])
        user_tf = count_vectorizer.transform([preprocessed_abstract])

        if lda_model is None:
            lda_model = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
            lda_top = lda_model.fit_transform(user_tf)
            st.write("Model LDA telah dilatih.")

        user_topic_distribution = lda_model.transform(user_tf)
        st.write(user_topic_distribution)
        y_pred = model2.predict(user_topic_distribution)
        y_pred
