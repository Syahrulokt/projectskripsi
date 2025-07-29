import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Fungsi untuk menampilkan tabel
def display_dataframe(df):
    st.write(df)

# Halaman beranda
def beranda():
    # Menampilkan judul dengan rata tengah
    st.markdown("""
        <h1 style="text-align: center; font-size: 50px">Sistem Klasterisasi UMKM di Kabupaten Sidoarjo menggunakan DBSCAN berbasis Perbandingan Jarak</h1>
    """, unsafe_allow_html=True)

    # Menambahkan garis
    st.markdown("""
        <hr style="border: 1px solid #DBDBDB; width: 100%; margin-top: 10px; margin-bottom: 20px;">
    """, unsafe_allow_html=True)

    # Menampilkan Gambar dari Folder ./dataset/
    image_path = 'umkm.jpg'
    st.image(image_path, use_container_width=True)

    # Menambahkan garis
    st.markdown("""
        <hr style="border: 1px solid #DBDBDB; width: 100%; margin-top: 10px; margin-bottom: 20px;">
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style="text-align: justify;">
            Selamat datang di aplikasi sistem klasterisasi UMKM di Kabupaten Sidoarjo.
            Project ini bertujuan untuk mengelompokkan Usaha Mikro, Kecil, dan Menengah (UMKM)
            menggunakan algoritma DBSCAN berbasis perbandingan jarak untuk membantu mengidentifikasi
            pola dan struktur dalam data UMKM yang ada di Sidoarjo.
        </p>
    """, unsafe_allow_html=True)

    # Langkah-langkah Penggunaan
    st.markdown("""
    <div style="background-color: #d0e7f7; font-size: 20px; padding-top: 7px; padding-bottom: 7px; padding-right: 7px; border-radius: 8px;">
        <strong>&nbsp;&nbsp;Langkah-langkah Penggunaan</strong>
    </div>
    """ , unsafe_allow_html=True)
    st.write("""
        1. **Masukkan Data UMKM**: Unggah file data UMKM yang akan digunakan.
        2. **Preprocessing Data**: Lakukan transformasi data, penanganan outlier, dan normalisasi data.
        3. **Modeling & Evaluasi**: Pilih jarak dan lakukan klasterisasi menggunakan algoritma DBSCAN.
        4. **Analisa Klaster**: Lihat hasil analisis data.
    """)
    st.write("")

    # Tentang Pengembang
    st.markdown("""
    <div style="background-color: #d0e7f7; font-size: 20px; padding-top: 7px; padding-bottom: 7px; padding-right: 7px; border-radius: 8px;">
        <strong>&nbsp;&nbsp;Tentang Pengembang</strong>
    </div>
    """ , unsafe_allow_html=True)
    st.write("""
        Aplikasi ini dikembangkan oleh Mochammad Syahrul Abidin, dibimbing oleh Dr. Yeni Kustiyahningsih, S.Kom., M.Kom 
        dan Eza Rahmanita, S.T., M.T. Tujuan project ini adalah untuk membantu UMKM di Sidoarjo dengan teknologi klasterisasi data.
    """)

# Halaman Masukkan Data
def upload_data():
    st.title("Masukkan Data")
    file = st.file_uploader("Pilih file .csv atau .xlsx", type=["csv", "xlsx"])

    if file is not None:
        try:
            # Coba membaca file CSV dengan pemisah titik koma dan encoding ISO-8859-1
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, sep=';', encoding='ISO-8859-1', on_bad_lines='skip')
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)

            # Menampilkan informasi dataset
            st.markdown("""
            <div style="background-color: #d0e7f7; font-size: 20px; padding-top: 7px; padding-bottom: 7px; padding-right: 7px; border-radius: 8px;">
                <strong>&nbsp;&nbsp;Informasi Dataset</strong>
            </div>
            """ , unsafe_allow_html=True)
            st.markdown("" , unsafe_allow_html=True)
            st.write(f"Jumlah Baris: {df.shape[0]} data")
            st.write(f"Jumlah Kolom: {df.shape[1]} kolom")
            
            # Menampilkan informasi kolom kategorikal dan numerik
            categorical_columns = df.select_dtypes(include=['object']).columns
            numerical_columns = df.select_dtypes(include=['number']).columns

            st.write(f"Jumlah Kolom Kategorikal: {len(categorical_columns)} kolom")
            st.write(f"Jumlah Kolom Numerik: {len(numerical_columns)} kolom")

            display_dataframe(df)
            
            # Menyimpan dataframe di session state untuk digunakan di halaman lain
            st.session_state.df = df
            return df
        except pd.errors.ParserError as e:
            st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    return None

def preprocessing_data():
    st.title("Preprocessing Data")

    # Pastikan bahwa data telah diupload
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu di menu 'Masukkan Data'.")
        return None  
    
    # Ambil data dari session state
    df = st.session_state.df

    # Sub-menu untuk memilih langkah preprocessing menggunakan radio buttons
    preprocess_option = st.radio(
        "Pilih Langkah Preprocessing",
        ["Transformasi Data", "Normalisasi Data"]
    )

    if preprocess_option == "Transformasi Data":
        st.write("### Transformasi Data")
        if st.button("Lakukan Transformasi Data"):
            # Memastikan data sudah diupload
            if 'df' not in st.session_state:
                st.warning("Silakan upload data terlebih dahulu di menu 'Masukkan Data'.")
                return None

            # Salin dataframe utama
            df = df.copy()
            st.session_state.original_df = df.copy()

            # Simpan salinan data sebelum transformasi
            df_before = df.copy()

            # Simpan nilai asli untuk ditampilkan di mapping
            izin_asli = df['IZIN USAHA'].copy()
            marketplace_asli = df['MARKETPLACE'].copy()

            # Bersihkan teks: ubah "Tidak ada" jadi "0"
            df['IZIN USAHA'] = df['IZIN USAHA'].replace("Tidak ada", "0")
            df['MARKETPLACE'] = df['MARKETPLACE'].replace("Tidak ada", "0")

            # Hitung jumlah entri pada setiap cell (NIB,IUMK → 2)
            df['IZIN USAHA'] = df['IZIN USAHA'].apply(lambda x: len(x.split(',')) if isinstance(x, str) and x != "0" else 0)
            df['MARKETPLACE'] = df['MARKETPLACE'].apply(lambda x: len(x.split(',')) if isinstance(x, str) and x != "0" else 0)

            # Label encode hasil jumlah tersebut
            label_encoder_izin = LabelEncoder()
            df['IZIN USAHA'] = label_encoder_izin.fit_transform(df['IZIN USAHA'])

            label_encoder_marketplace = LabelEncoder()
            df['MARKETPLACE'] = label_encoder_marketplace.fit_transform(df['MARKETPLACE'])

            # Buat mapping (data asli sebagai before, label encoded sebagai after)
            izin_mapping_df = pd.DataFrame({
                'IZIN USAHA_before': izin_asli,
                'IZIN USAHA_after': df['IZIN USAHA']
            }).drop_duplicates(subset='IZIN USAHA_after').sort_values(by='IZIN USAHA_after')  # Hanya tampilkan after yang unik

            marketplace_mapping_df = pd.DataFrame({
                'MARKETPLACE_before': marketplace_asli,
                'MARKETPLACE_after': df['MARKETPLACE']
            }).drop_duplicates(subset='MARKETPLACE_after').sort_values(by='MARKETPLACE_after')  # Hanya tampilkan after yang unik

            # Simpan ke session state
            st.session_state.df = df
            st.session_state.df_before = df_before

            # Tampilkan data setelah transformasi
            st.markdown("""
                <div style="background-color: #d0e7f7; font-size: 15px; padding-top: 7px; padding-bottom: 7px; padding-right: 7px; border-radius: 8px;">
                <strong>&nbsp;&nbsp;Data setelah Label Encoding</strong>
                </div>
                """, unsafe_allow_html=True)
            st.write("")
            st.dataframe(df)
                        
            # Tampilkan mapping di dalam expander
            with st.expander("Lihat Mapping Label Encoding"):
                st.write("**IZIN USAHA (unik berdasarkan hasil encoding):**")
                st.dataframe(izin_mapping_df)

                st.write("**MARKETPLACE (unik berdasarkan hasil encoding):**")
                st.dataframe(marketplace_mapping_df)

    # Jika user memilih Normalisasi Data
    elif preprocess_option == "Normalisasi Data":
        st.write("### Normalisasi Data")
        if st.button("Lakukan Normalisasi MaxAbs"):
            # Memastikan data sudah diupload
            if 'df' not in st.session_state:
                st.warning("Silakan upload data terlebih dahulu di menu 'Masukkan Data'.")
                return None

            # Ambil data dari session state
            df = st.session_state.df

            # Pilih kolom numerik untuk normalisasi
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Normalisasi data menggunakan MaxAbs untuk Euclidean
            df_euclidean = df.copy()
            for col in numerical_columns:
                if col in df_euclidean.columns:
                    df_euclidean[col] = df_euclidean[col] / df_euclidean[col].max()

            # Menyimpan hasil normalisasi Euclidean di session state
            st.session_state.df_euclidean = df_euclidean

            # Normalisasi data menggunakan MaxAbs untuk Manhattan
            df_manhattan = df.copy()
            for col in numerical_columns:
                if col in df_manhattan.columns:
                    df_manhattan[col] = df_manhattan[col] / df_manhattan[col].max()

            # Menyimpan hasil normalisasi Manhattan di session state
            st.session_state.df_manhattan = df_manhattan

            # Normalisasi data menggunakan MaxAbs untuk Hamming
            df_hamming = df.copy()
            for col in numerical_columns:
                if col in df_hamming.columns:
                    df_hamming[col] = df_hamming[col] / df_hamming[col].max()

            # Menyimpan hasil normalisasi Hamming di session state
            st.session_state.df_hamming = df_hamming

            # Menampilkan data setelah normalisasi
            st.markdown("""
                <div style="background-color: #d0e7f7; font-size: 15px; padding-top: 7px; padding-bottom: 7px; padding-right: 7px; border-radius: 8px;">
                   <strong>&nbsp;&nbsp;Data setelah normalisasi MaxAbs</strong>
                </div>
                """ , unsafe_allow_html=True)
            st.write("")
            display_dataframe(df_euclidean)

            # Menyimpan data yang sudah dinormalisasi pada session state
            st.session_state.df_euclidean = df_euclidean
            st.session_state.df_manhattan = df_manhattan
            st.session_state.df_hamming = df_hamming

        return df

def modeling(df, metric):
    st.title(f"Modeling - DBSCAN Clustering dengan Jarak {metric.capitalize()}")

    # Memastikan data sudah ada di session state
    if 'df_euclidean' not in st.session_state or 'df_manhattan' not in st.session_state or 'df_hamming' not in st.session_state:
        st.warning("Silakan lakukan normalisasi terlebih dahulu")
        return None
    
    # Pilih data berdasarkan normalisasi yang dipilih
    if metric == "euclidean":
        df_normalized = st.session_state.df_euclidean
    elif metric == "manhattan":
        df_normalized = st.session_state.df_manhattan
    elif metric == "hamming":
        df_normalized = st.session_state.df_hamming
    else:
        st.warning("Pilih jenis normalisasi yang valid.")
        return None

    # Pilih hanya kolom numerik untuk clustering
    df_numeric = df_normalized.select_dtypes(include=[np.number])

    if df_numeric.empty:
        st.error("Tidak ada kolom numerik yang tersedia untuk clustering.")
        return None

    # Input parameter DBSCAN
    eps = st.slider("Pilih nilai eps", 0.1, 1.0, 0.2)
    min_samples = st.slider("Pilih nilai minPts", 1, 100, 49)

    # Melakukan clustering dengan DBSCAN
    if st.button(f"Lakukan Clustering menggunakan {metric} distance"):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric.lower())
        clusters = dbscan.fit_predict(df_numeric)

        # Menambahkan hasil cluster ke data dengan nama kolom sesuai dengan jarak
        cluster_column_name = f"CLUSTER_{metric.upper()}"
        df_normalized[cluster_column_name] = clusters

        # Menyimpan hasil clustering dalam session state dengan nama yang sesuai dengan jarak
        st.session_state[f"df_{metric}"] = df_normalized 

        # Menampilkan hasil clustering
        st.markdown(f"""
            <div style="background-color: #d0e7f7; font-size: 15px; padding-top: 7px; padding-bottom: 7px; padding-right: 7px; border-radius: 8px;">
                <strong>&nbsp;&nbsp;Hasil Clustering menggunakan {metric} distance</strong>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        display_dataframe(df_normalized)

        # Memisahkan dan menampilkan setiap cluster
        unique_clusters = sorted(df_normalized[cluster_column_name].unique())
        for cluster in unique_clusters:
            st.write(f"### Cluster {cluster}")
            cluster_data = df_normalized[df_normalized[cluster_column_name] == cluster]
            st.markdown(f"""
                <div style="background-color: #d0e7f7; font-size: 15px; padding-top: 7px; padding-bottom: 7px; padding-right: 7px; border-radius: 8px;">
                    <strong>&nbsp;&nbsp;Jumlah data pada Cluster {cluster} sebanyak {cluster_data.shape[0]} data</strong>
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.write(cluster_data) 

        # Visualisasi hasil t-SNE
        tsne = TSNE(n_components=2, perplexity=60, n_iter=1000, metric=metric.lower(), random_state=42)
        tsne_result = tsne.fit_transform(df_normalized.select_dtypes(include=[np.number]))

        # Masukkan hasil ke DataFrame untuk t-SNE
        df_tsne = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
        df_tsne['Cluster'] = df_normalized[cluster_column_name]

        # Plot t-SNE
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='TSNE1', y='TSNE2',
            hue='Cluster',
            palette='tab10',
            data=df_tsne,
            alpha=0.7
        )
        plt.title(f'Visualisasi 2D Clustering DBSCAN ({metric.capitalize()} Distance) dengan t-SNE')
        plt.xlabel('t-SNE Komponen 1')
        plt.ylabel('t-SNE Komponen 2')
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

        return df_normalized
    return None

# Fungsi untuk mengevaluasi model
def evaluate_model(df, metric):
    st.title(f"Evaluasi Hasil Clustering ({metric.capitalize()})")
    
    if df is not None:
        # Menentukan nama kolom cluster berdasarkan metric
        cluster_column = f"CLUSTER_{metric.upper()}"

        # Memastikan kolom cluster ada di dataframe
        if cluster_column not in df.columns:
            st.error(f"Kolom {cluster_column} tidak ada. Silakan lakukan clustering terlebih dahulu.")
            return None
        
        # Memfilter data untuk menghindari outlier (cluster == -1)
        df_valid = df[df[cluster_column] != -1]  # Menghapus baris yang memiliki cluster -1 (outliers)
        
        if df_valid.empty:
            st.error("Semua data merupakan outlier (cluster -1). Tidak dapat dilakukan evaluasi.")
            return None
        
        # Mengecek jumlah cluster valid
        unique_clusters = df_valid[cluster_column].nunique()

        # Peringatan jika hanya ada 1 cluster
        if unique_clusters == 1:
            st.warning("Peringatan: Hanya ada 1 cluster yang terbentuk. Evaluasi SC dan DBI tidak valid karena tidak ada variasi dalam cluster.")
            return None

        # Menampilkan jumlah data yang dievaluasi
        num_valid_data = df_valid.shape[0]
        st.markdown(f"""
                <div style="background-color: #d0e7f7; font-size: 15px; padding-top: 7px; padding-bottom: 7px; padding-right: 7px; border-radius: 8px;">
                    <strong>&nbsp;&nbsp;Jumlah data yang dievaluasi adalah {num_valid_data} data</strong>
                </div>
            """, unsafe_allow_html=True)
        st.write("")

        # Mengambil kolom numerik untuk evaluasi
        df_numeric = df_valid.select_dtypes(include=[np.number])

        # Evaluasi Silhouette Score dan Davies-Bouldin Index
        silhouette_avg = silhouette_score(df_numeric, df_valid[cluster_column])
        dbi = davies_bouldin_score(df_numeric, df_valid[cluster_column])

        st.write(f"Nilai Silhouette Coefficient (SC) untuk {cluster_column} yaitu {silhouette_avg:.2f}")
        st.write(f"Davies-Bouldin Index (DBI) untuk {cluster_column} yaitu {dbi:.2f}")

        # Tampilkan grafik perbandingan
        fig, ax = plt.subplots()
        ax.bar(['Silhouette Coefficient', 'Davies-Bouldin Index'], [silhouette_avg, dbi])
        ax.set_ylabel('Score')
        st.pyplot(fig)

        st.title("Hasil Clustering")
    
        # Mengambil data asli dari session state
        if 'original_df' not in st.session_state:
            st.warning("Silakan upload dan proses data terlebih dahulu.")
            return

        # Mengambil data asli (sebelum diproses)
        original_df = st.session_state.original_df

        # Memastikan bahwa kolom cluster ada
        if cluster_column in df.columns:
            # Menambahkan kolom cluster ke data asli
            original_df[cluster_column] = df[cluster_column]

            # Menyimpan data asli yang sudah memiliki kolom cluster dengan nama variabel
            cluster_data_var_name = f"CLUSTER_{metric.upper()}"
            st.session_state[cluster_data_var_name] = original_df

            st.markdown("""
                <div style="background-color: #d0e7f7; font-size: 15px; padding-top: 7px; padding-bottom: 7px; padding-right: 7px; border-radius: 8px;">
                    <strong>&nbsp;&nbsp;Data keseluruhan hasil Clustering</strong>
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.write(original_df)  # Menampilkan data asli dengan kolom cluster
        
        else:
            st.write("Silakan lakukan clustering terlebih dahulu.")

def analyze_clusters(df, metric):
    st.markdown(f"""
                <div style="background-color: #d0e7f7; font-size: 17px; padding-top: 7px; padding-bottom: 7px; padding-right: 7px; border-radius: 8px;">
                    <strong>&nbsp;&nbsp;Analisa setiap Clustering pada Algoritma DBSCAN pada {metric.capitalize()} Distance</strong>
                </div>
            """, unsafe_allow_html=True)
    st.write("")
    # Memastikan data asli dengan hasil clustering sudah ada di session state
    if 'original_df' not in st.session_state:
        st.write("Data asli dengan hasil clustering tidak ditemukan. Silakan lakukan clustering terlebih dahulu.")
        return

    # Ambil data asli dengan hasil clustering
    df_relevant = st.session_state.original_df

    cluster_column = f"CLUSTER_{metric.upper()}"

    # Memastikan kolom cluster sesuai metric ada di dataframe
    if cluster_column not in df_relevant.columns:
        st.warning(f"{cluster_column} tidak ditemukan di data. Silakan pastikan clustering dilakukan terlebih dahulu.")
        return
    
     # Memisahkan data berdasarkan cluster, namun mengabaikan cluster -1 (outliers)
    df_relevant = df_relevant[df_relevant[cluster_column] != -1]  # Menghilangkan outliers (Cluster -1)
    unique_clusters = sorted(df_relevant[cluster_column].unique())
    
    # Fitur untuk dianalisis
    features = [
        "MODAL",
        "OMSET PER BULAN",
        "TENAGA KERJA",
        "IZIN USAHA",
        "MARKETPLACE",
        "JENIS USAHA",
        "KECAMATAN"
    ]
    
    # Siapkan struktur untuk tabel analisis
    analysis_table = []

    for cluster in unique_clusters:
        cluster_data = df_relevant[df_relevant[cluster_column] == cluster]
        
        # Siapkan baris untuk kluster saat ini
        row = {'CLUSTER': f"CLUSTER {cluster}"}
        
        # Untuk setiap fitur, hitung Min, Max, dan Rata-rata
        for feature in features:
            if feature in cluster_data.columns:
                # Jika fitur bersifat numerik, hitung min, max, dan rata-rata
                if cluster_data[feature].dtype in ['float64', 'int64']:
                    row[f"{feature} MINIMAL"] = cluster_data[feature].min()
                    row[f"{feature} MAKSIMAL"] = cluster_data[feature].max()
                    row[f"{feature} RATA-RATA"] = cluster_data[feature].mean()
                else:
                    # Jika fitur bersifat kategorikal, hitung nilai-nilai terbanyak (seperti KECAMATAN)
                    row[f"{feature} TERBANYAK"] = ", ".join(cluster_data[feature].value_counts().head(1).index)

        # Tambahkan baris ke tabel analisis
        analysis_table.append(row)

    # Mengubah tabel analisis menjadi DataFrame
    analysis_df = pd.DataFrame(analysis_table)

    # Menerapkan beberapa format untuk kejelasan tampilan
    formatted_df = analysis_df.copy()

    # Mengubah format angka untuk kolom yang jenisnya numerik
    for col in analysis_df.columns:
        if 'MINIMAL' in col or 'MAKSIMAL' in col or 'RATA-RATA' in col:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.0f}".replace(',', '.') if isinstance(x, (int, float)) else x)

    st.write(formatted_df)

def main():
    st.sidebar.title("Sistem Clustering UMKM")
    menu = [
        "Beranda",
        "Masukkan Data",
        "Preprocessing Data",
        "Modeling dan Evaluasi DBSCAN Clustering dengan Jarak Euclidean",
        "Modeling dan Evaluasi DBSCAN Clustering dengan Jarak Manhattan",
        "Modeling dan Evaluasi DBSCAN Clustering dengan Jarak Hamming",
        "Analisa Setiap Clustering"
    ]
    choice = st.sidebar.radio("Pilih Menu", menu)

    if choice == "Beranda":
        beranda()
    elif choice == "Masukkan Data":
        df = upload_data()
    elif choice == "Preprocessing Data":
        df = preprocessing_data()
    elif choice == "Modeling dan Evaluasi DBSCAN Clustering dengan Jarak Euclidean":
        if 'df' in st.session_state:
            df = modeling(st.session_state.df, metric="euclidean")
            evaluate_model(df, metric="euclidean")
        else:
            st.warning("Silakan upload dan proses data terlebih dahulu.")
    elif choice == "Modeling dan Evaluasi DBSCAN Clustering dengan Jarak Manhattan":
        if 'df' in st.session_state:
            df = modeling(st.session_state.df, metric="manhattan")
            evaluate_model(df, metric="manhattan")
        else:
            st.warning("Silakan upload dan proses data terlebih dahulu.")
    elif choice == "Modeling dan Evaluasi DBSCAN Clustering dengan Jarak Hamming":
        if 'df' in st.session_state:
            df = modeling(st.session_state.df, metric="hamming")
            evaluate_model(df, metric="hamming")
        else:
            st.warning("Silakan upload dan proses data terlebih dahulu.")
    # elif choice == "Metode Klasifikasi":
    #     evaluate_classification(st.session_state.df)
    elif choice == "Analisa Setiap Clustering":
        if 'df' in st.session_state:
            metric = st.selectbox("Pilih Jarak untuk Analisa", ["euclidean", "manhattan", "hamming"])
            analyze_clusters(st.session_state.df, metric)
        else:
            st.warning("Silakan lakukan clustering terlebih dahulu.")

# Mengubah warna tombol setelah ditekan
st.markdown("""<style> .stButton>button {background-color: #FFBD73;} </style>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
