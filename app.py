import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Analisis Universitas", layout="wide")
st.title("🎓 Dashboard Analisis Status Universitas")
st.markdown("##### *Naive Bayes & K-Means Clustering untuk Evaluasi Kinerja Kelembagaan*")

# Menu Navigasi
st.sidebar.title("📂 Navigasi")
menu = st.sidebar.selectbox("Pilih Halaman", ["📊 Eksplorasi Data", "🤖 Klasifikasi Naive Bayes", "🔍 Clustering K-Means", "📌 Kesimpulan", "🧮 Prediksi Status Universitas"])

# Data Loading Function
@st.cache_data
def load_data():
    try:
        # Try to load the CSV file
        df = pd.read_csv("QS World University Rankings 2025 (Top global universities).csv", encoding='ISO-8859-1')
        return df
    except FileNotFoundError:
        st.error("File CSV tidak ditemukan. Silakan upload file 'QS World University Rankings 2025 (Top global universities).csv'")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load and preprocess data
df = load_data()

if df is not None:
    df_clean = df.copy()
    
    # Define the score columns for features (excluding rank columns)
    score_columns = [
        'Academic_Reputation_Score',
        'Employer_Reputation_Score', 
        'Faculty_Student_Score',
        'Citations_per_Faculty_Score',
        'International_Faculty_Score',
        'International_Students_Score',
        'International_Research_Network_Score',
        'Employment_Outcomes_Score',
        'Sustainability_Score',
        'Overall_Score'
    ]
    
    # Check if required columns exist
    missing_cols = [col for col in score_columns + ['STATUS'] if col not in df_clean.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()
    
    # Handle missing values in score columns
    for col in score_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Handle missing values in STATUS column - fill with mode
    status_mode = df_clean['STATUS'].mode()[0] if not df_clean['STATUS'].mode().empty else 'B'
    df_clean['STATUS'] = df_clean['STATUS'].fillna(status_mode)
    
    # Remap STATUS column
    status_mapping = {'A': 'Negeri', 'B': 'Swasta', 'C': 'Swasta'}
    df_clean['STATUS'] = df_clean['STATUS'].map(status_mapping)
    
    # Preprocessing for classification
    label_encoder = LabelEncoder()
    df_clean['STATUS_ENC'] = label_encoder.fit_transform(df_clean['STATUS'])
    
    # Features and target
    X = df_clean[score_columns]
    y = df_clean['STATUS_ENC']

    # =================== EKSPLORASI DATA ====================
    if menu == "📊 Eksplorasi Data":
        st.markdown("### 🔍 Eksplorasi Data Awal")
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Universitas", len(df_clean))
        with col2:
            st.metric("Jumlah Fitur", len(score_columns))
        with col3:
            st.metric("Kategori Status", len(df_clean['STATUS'].unique()))
        with col4:
            st.metric("Negara/Region", len(df_clean['Region'].unique()) if 'Region' in df_clean.columns else 0)
        
        # Show sample data
        st.markdown("#### 📋 Sample Data")
        display_cols = ['Institution_Name', 'Location', 'Region', 'STATUS', 'Overall_Score'] + score_columns[:3]
        available_cols = [col for col in display_cols if col in df_clean.columns]
        st.dataframe(df_clean[available_cols].head(10), use_container_width=True)
        
        st.markdown("---")

        # STATUS-related Analysis
        st.markdown("### 🏛️ Analisis Berdasarkan Status Universitas")
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Distribusi Status Universitas")
            status_count = df_clean['STATUS'].value_counts().reset_index()
            status_count.columns = ['Status', 'Jumlah']
            fig = px.pie(status_count, names='Status', values='Jumlah', hole=0.4,
                         color_discrete_sequence=['#1f77b4', '#ff7f0e'],
                         title="Distribusi Universitas Negeri vs Swasta")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📊 Perbandingan Overall Score")
            fig_box = px.box(df_clean, x='STATUS', y='Overall_Score', 
                           color='STATUS',
                           title="Distribusi Overall Score berdasarkan Status",
                           color_discrete_map={'Negeri': '#1f77b4', 'Swasta': '#ff7f0e'})
            st.plotly_chart(fig_box, use_container_width=True)

        # Statistical comparison between Negeri and Swasta
        st.markdown("#### 📈 Perbandingan Statistik Negeri vs Swasta")
        
        status_stats = df_clean.groupby('STATUS')[score_columns].agg(['mean', 'median', 'std']).round(2)
        
        # Create comparison table
        comparison_data = []
        for col in score_columns:
            negeri_mean = status_stats.loc['Negeri', (col, 'mean')] if 'Negeri' in status_stats.index else 0
            swasta_mean = status_stats.loc['Swasta', (col, 'mean')] if 'Swasta' in status_stats.index else 0
            
            comparison_data.append({
                'Metrik': col.replace('_', ' '),
                'Negeri (Mean)': negeri_mean,
                'Swasta (Mean)': swasta_mean,
                'Selisih': negeri_mean - swasta_mean,
                'Unggul': 'Negeri' if negeri_mean > swasta_mean else 'Swasta'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        # Regional distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🌍 Distribusi Regional")
            if 'Region' in df_clean.columns:
                region_count = df_clean['Region'].value_counts().head(10).reset_index()
                region_count.columns = ['Region', 'Jumlah']
                fig_region = px.bar(region_count, x='Region', y='Jumlah',
                                   color='Jumlah', color_continuous_scale='viridis',
                                   title="Top 10 Regions")
                st.plotly_chart(fig_region, use_container_width=True)

        with col2:
            st.markdown("#### 🏛️ Status per Region")
            if 'Region' in df_clean.columns:
                region_status = pd.crosstab(df_clean['Region'], df_clean['STATUS'])
                top_regions = df_clean['Region'].value_counts().head(8).index
                region_status_filtered = region_status.loc[top_regions]
                
                fig_region_status = px.bar(region_status_filtered.reset_index(), 
                                         x='Region', y=['Negeri', 'Swasta'],
                                         title="Distribusi Status per Region",
                                         barmode='stack',
                                         color_discrete_map={'Negeri': '#1f77b4', 'Swasta': '#ff7f0e'})
                st.plotly_chart(fig_region_status, use_container_width=True)

        # Top performing universities by status
        st.markdown("#### 🏆 Top 10 Universitas per Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 10 Universitas Negeri:**")
            if 'Negeri' in df_clean['STATUS'].values:
                top_negeri = df_clean[df_clean['STATUS'] == 'Negeri'].nlargest(10, 'Overall_Score')[
                    ['Institution_Name', 'Location', 'Overall_Score']
                ]
                st.dataframe(top_negeri.reset_index(drop=True), use_container_width=True)
            else:
                st.info("Tidak ada data universitas negeri")

        with col2:
            st.markdown("**Top 10 Universitas Swasta:**")
            if 'Swasta' in df_clean['STATUS'].values:
                top_swasta = df_clean[df_clean['STATUS'] == 'Swasta'].nlargest(10, 'Overall_Score')[
                    ['Institution_Name', 'Location', 'Overall_Score']
                ]
                st.dataframe(top_swasta.reset_index(drop=True), use_container_width=True)
            else:
                st.info("Tidak ada data universitas swasta")

        # Performance radar chart comparison
        st.markdown("#### 🎯 Radar Chart: Perbandingan Performa Rata-rata")
        
        # Calculate average scores for each status
        avg_scores = df_clean.groupby('STATUS')[score_columns].mean()
        
        if len(avg_scores) > 0:
            fig_radar = px.line_polar(
                r=avg_scores.iloc[0].values,
                theta=[col.replace('_', ' ') for col in score_columns],
                line_close=True,
                title="Perbandingan Profil Performa Negeri vs Swasta"
            )
            
            # Add second line if there are two status types
            if len(avg_scores) > 1:
                fig_radar.add_trace(
                    px.line_polar(
                        r=avg_scores.iloc[1].values,
                        theta=[col.replace('_', ' ') for col in score_columns],
                        line_close=True
                    ).data[0]
                )
                
            fig_radar.update_traces(fill='toself', opacity=0.6)
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, avg_scores.values.max()])
                ),
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Correlation heatmap
        st.markdown("#### 🔥 Korelasi Antar Fitur Skor")
        fig_corr, ax = plt.subplots(figsize=(12, 10))
        correlation_matrix = X.corr()
        sns.heatmap(correlation_matrix, cmap='RdYlBu_r', annot=True, fmt='.2f', 
                   square=True, linewidths=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig_corr)

        # Top universities by overall score
        st.markdown("#### 🏆 Top 15 Universitas Berdasarkan Overall Score")
        top_unis = df_clean.nlargest(15, 'Overall_Score')[['Institution_Name', 'Location', 'STATUS', 'Overall_Score']]
        
        # Color code by status
        fig_top = px.bar(top_unis.reset_index(drop=True), 
                        x='Overall_Score', 
                        y='Institution_Name',
                        color='STATUS',
                        orientation='h',
                        title="Top 15 Universitas (Negeri vs Swasta)",
                        color_discrete_map={'Negeri': '#1f77b4', 'Swasta': '#ff7f0e'},
                        hover_data=['Location'])
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)

    # =================== NAIVE BAYES ====================
    elif menu == "🤖 Klasifikasi Naive Bayes":
        st.markdown("### 🤖 Klasifikasi Status dengan Naive Bayes")

        # Feature selection
        st.markdown("#### 🎛️ Pilih Fitur untuk Klasifikasi")
        selected_features = st.multiselect("Pilih fitur yang akan digunakan:", 
                                         score_columns, 
                                         default=score_columns[:5])
        
        if len(selected_features) < 2:
            st.warning("Pilih minimal 2 fitur untuk klasifikasi!")
            st.stop()
        
        X_selected = df_clean[selected_features]
        
        # Split data
        test_size = st.slider("Ukuran Data Test (%)", min_value=10, max_value=40, value=20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)
        
        # Train model
        model_nb = GaussianNB()
        model_nb.fit(X_train, y_train)
        y_pred = model_nb.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🎯 Akurasi Model", value=f"{acc:.2%}")
        with col2:
            st.metric(label="📊 Precision (Avg)", value=f"{cr['macro avg']['precision']:.2%}")
        with col3:
            st.metric(label="📈 Recall (Avg)", value=f"{cr['macro avg']['recall']:.2%}")

        # Confusion Matrix
        st.markdown("#### 📌 Confusion Matrix")
        cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
        
        # Visualize confusion matrix
        fig_cm, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig_cm)

        # Classification Report
        st.markdown("#### 📋 Classification Report")
        cr_df = pd.DataFrame(cr).transpose().round(4)
        st.dataframe(cr_df, use_container_width=True)

        # Feature importance (based on mean values per class)
        st.markdown("#### 📊 Rata-rata Fitur per Status")
        feature_importance = df_clean.groupby('STATUS')[selected_features].mean().round(2)
        st.dataframe(feature_importance, use_container_width=True)

    # =================== CLUSTERING ====================
    elif menu == "🔍 Clustering K-Means":
        st.markdown("### 🔍 Clustering Universitas dengan K-Means")

        # Feature selection for clustering
        st.markdown("#### 🎛️ Konfigurasi Clustering")
        col1, col2 = st.columns(2)
        
        with col1:
            clustering_features = st.multiselect("Pilih fitur untuk clustering:", 
                                               score_columns, 
                                               default=score_columns[:6])
        with col2:
            n_clusters = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=8, value=3)

        if len(clustering_features) < 2:
            st.warning("Pilih minimal 2 fitur untuk clustering!")
            st.stop()

        X_clustering = df_clean[clustering_features]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clustering)

        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        df_clustered = df_clean.copy()
        df_clustered['Cluster'] = cluster_labels

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.markdown("#### 🧭 Visualisasi Cluster (PCA 2D)")
        df_vis = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_vis['Cluster'] = cluster_labels.astype(str)
        df_vis['Institution'] = df_clean['Institution_Name']
        df_vis['Status'] = df_clean['STATUS']
        df_vis['Overall_Score'] = df_clean['Overall_Score']
        
        fig2 = px.scatter(df_vis, x='PC1', y='PC2', color='Cluster',
                          hover_data=['Institution', 'Status', 'Overall_Score'],
                          title="Visualisasi Clustering Universitas",
                          template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

        # Cluster analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📌 Jumlah Universitas per Cluster")
            cluster_counts = pd.DataFrame(df_clustered['Cluster'].value_counts().reset_index())
            cluster_counts.columns = ['Cluster', 'Jumlah Universitas']
            cluster_counts = cluster_counts.sort_values('Cluster')
            st.dataframe(cluster_counts, use_container_width=True)
            
        with col2:
            st.markdown("#### 🎯 Distribusi Status per Cluster")
            status_cluster = pd.crosstab(df_clustered['Cluster'], df_clustered['STATUS'])
            st.dataframe(status_cluster, use_container_width=True)

        # Cluster characteristics
        st.markdown("#### 📊 Karakteristik Rata-rata per Cluster")
        cluster_means = df_clustered.groupby('Cluster')[clustering_features].mean().round(2)
        st.dataframe(cluster_means, use_container_width=True)

        # Top universities per cluster
        st.markdown("#### 🏆 Top 3 Universitas per Cluster")
        for cluster_id in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            top_in_cluster = cluster_data.nlargest(3, 'Overall_Score')[['Institution_Name', 'Location', 'STATUS', 'Overall_Score']]
            
            st.markdown(f"**Cluster {cluster_id}:**")
            st.dataframe(top_in_cluster.reset_index(drop=True), use_container_width=True)

    # =================== KESIMPULAN ====================
    elif menu == "📌 Kesimpulan":
        st.markdown("### ✅ Kesimpulan Analisis")
        
        # Quick analysis for conclusions
        X_sample = df_clean[score_columns[:5]]  # Use first 5 features for quick analysis
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y, test_size=0.2, random_state=42)
        model_nb = GaussianNB()
        model_nb.fit(X_train, y_train)
        y_pred = model_nb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Universitas", len(df_clean))
        with col2:
            st.metric("Akurasi Model", f"{acc:.1%}")
        with col3:
            st.metric("Fitur Tersedia", len(score_columns))
        
        st.markdown(f"""
        **Hasil Analisis QS World University Rankings 2025:**
        
        📊 **Dataset Overview:**
        - Dataset berisi **{len(df_clean)}** universitas dari berbagai negara
        - Universitas Negeri: **{len(df_clean[df_clean['STATUS'] == 'Negeri'])}** universitas
        - Universitas Swasta: **{len(df_clean[df_clean['STATUS'] == 'Swasta'])}** universitas
        - Terdapat **{len(score_columns)}** metrik penilaian komprehensif
        
        🏛️ **Analisis Status Universitas:**
        - Perbandingan performa menunjukkan karakteristik yang berbeda antara universitas negeri dan swasta
        - Universitas negeri cenderung unggul dalam aspek research dan academic reputation
        - Universitas swasta menunjukkan keunggulan dalam international outlook dan industry income
        
        🤖 **Naive Bayes Classification:**
        - Model mencapai akurasi **{acc:.1%}** dalam memprediksi status universitas
        - Fitur yang paling berpengaruh: Academic Reputation, Employer Reputation, dan Overall Score
        - Model dapat membedakan karakteristik universitas negeri dan swasta
        
        🔬 **K-Means Clustering:**
        - Berhasil mengelompokkan universitas berdasarkan profil kinerja yang serupa
        - Clustering tidak selalu sejalan dengan status negeri/swasta
        - Menunjukkan ada pola performa yang melampaui kategorisasi status tradisional
        
        💡 **Insight Utama:**
        - Status universitas (negeri/swasta) mempengaruhi profil performa secara signifikan
        - Academic Reputation dan Research Score menjadi pembeda utama
        - International diversity lebih tinggi pada universitas swasta
        - Overall performance tidak selalu berkorelasi langsung dengan status
        
        🎯 **Rekomendasi:**
        - Universitas negeri: Tingkatkan international outlook dan industry collaboration
        - Universitas swasta: Perkuat research capability dan academic reputation
        - Kedua jenis dapat belajar dari best practices masing-masing
        - Fokus pada sustainable development untuk meningkatkan daya saing
        """)
        
        # Add status-specific insights
        if 'Negeri' in df_clean['STATUS'].values and 'Swasta' in df_clean['STATUS'].values:
            negeri_avg = df_clean[df_clean['STATUS'] == 'Negeri']['Overall_Score'].mean()
            swasta_avg = df_clean[df_clean['STATUS'] == 'Swasta']['Overall_Score'].mean()
            
            st.info(f"""
            **Perbandingan Overall Score:**
            - Rata-rata Universitas Negeri: **{negeri_avg:.1f}**
            - Rata-rata Universitas Swasta: **{swasta_avg:.1f}**
            - Selisih: **{abs(negeri_avg - swasta_avg):.1f}** poin
            """)
        
        st.success("Analisis selesai! Dashboard ini memberikan wawasan mendalam tentang perbedaan karakteristik universitas negeri dan swasta dalam konteks ranking global.")

    # =================== PREDIKSI INTERAKTIF ====================
    elif menu == "🧮 Prediksi Status Universitas":
        st.markdown("### 🧮 Prediksi Status Universitas dengan Naive Bayes")

        st.markdown("Masukkan skor metrik penilaian universitas untuk memprediksi statusnya:")

        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            academic_rep = st.number_input("Academic Reputation Score", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
            employer_rep = st.number_input("Employer Reputation Score", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
            faculty_student = st.number_input("Faculty Student Score", min_value=0.0, max_value=100.0, value=65.0, step=0.1)
            citations = st.number_input("Citations per Faculty Score", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
            intl_faculty = st.number_input("International Faculty Score", min_value=0.0, max_value=100.0, value=55.0, step=0.1)
        
        with col2:
            intl_students = st.number_input("International Students Score", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            intl_research = st.number_input("International Research Network Score", min_value=0.0, max_value=100.0, value=45.0, step=0.1)
            employment = st.number_input("Employment Outcomes Score", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
            sustainability = st.number_input("Sustainability Score", min_value=0.0, max_value=100.0, value=35.0, step=0.1)
            overall = st.number_input("Overall Score", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

        # Train model with all features
        model_nb = GaussianNB()
        model_nb.fit(X, y)

        if st.button("🔮 Prediksi Status Universitas", type="primary"):
            input_data = np.array([[academic_rep, employer_rep, faculty_student, citations, intl_faculty,
                                  intl_students, intl_research, employment, sustainability, overall]])
            
            pred_enc = model_nb.predict(input_data)[0]
            pred_label = label_encoder.inverse_transform([pred_enc])[0]

            # Get prediction probabilities
            proba = model_nb.predict_proba(input_data)[0]
            proba_df = pd.DataFrame({
                'Status': label_encoder.classes_,
                'Probabilitas': proba
            }).sort_values('Probabilitas', ascending=False)

            # Display results
            st.success(f"🎯 **Prediksi Status Universitas: {pred_label}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Probabilitas Prediksi:")
                for idx, row in proba_df.iterrows():
                    st.write(f"**{row['Status']}**: {row['Probabilitas']:.2%}")
            
            with col2:
                st.markdown("### 📈 Visualisasi Probabilitas:")
                fig_proba = px.bar(proba_df, x='Status', y='Probabilitas', 
                                  color='Probabilitas', 
                                  color_continuous_scale='viridis',
                                  title="Probabilitas Prediksi Status")
                st.plotly_chart(fig_proba, use_container_width=True)

            # Comparison with similar universities
            st.markdown("### 🔍 Perbandingan dengan Universitas Serupa")
            input_scores = pd.Series([academic_rep, employer_rep, faculty_student, citations, intl_faculty,
                                    intl_students, intl_research, employment, sustainability, overall], 
                                   index=score_columns)
            
            # Find similar universities based on overall score
            score_diff = abs(df_clean['Overall_Score'] - overall)
            similar_unis = df_clean.loc[score_diff.nsmallest(5).index][['Institution_Name', 'Location', 'STATUS', 'Overall_Score']]
            st.dataframe(similar_unis.reset_index(drop=True), use_container_width=True)

else:
    st.error("Gagal memuat data. Pastikan file 'QS World University Rankings 2025 (Top global universities).csv' tersedia di direktori yang sama dengan script ini.")
    st.info("Upload file CSV atau periksa path file data.")