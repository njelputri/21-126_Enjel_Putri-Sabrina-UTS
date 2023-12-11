import pickle
import librosa
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import skew, kurtosis, mode, iqr
from sklearn.model_selection import train_test_split


st.set_page_config(layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>Klasifikasi Data Audio UTS</h1>", unsafe_allow_html=True
)


def calculate_statistics(audio_path):
    x, sr = librosa.load(audio_path)

    mean = np.mean(x)
    std = np.std(x)
    maxv = np.amax(x)
    minv = np.amin(x)
    median = np.median(x)
    skewness = skew(x)
    kurt = kurtosis(x)
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    mode_v = mode(x)[0]
    iqr = q3 - q1

    zcr = librosa.feature.zero_crossing_rate(x) 
    mean_zcr = np.mean(zcr)
    median_zcr = np.median(zcr)
    std_zcr = np.std(zcr)
    skew_zcr = skew(zcr, axis=None)
    kurtosis_zcr = kurtosis(zcr, axis=None)

    n = len(x)
    mean_rms = np.sqrt(np.mean(x**2) / n)
    median_rms = np.sqrt(np.median(x**2) / n)
    std_rms = np.sqrt(np.std(x**2) / n)
    skew_rms = np.sqrt(skew(x**2) / n)
    kurtosis_rms = np.sqrt(kurtosis(x**2) / n)

    return [mean, median, mode_v, maxv, minv, std, skewness, kurt, q1, q3, iqr, mean_zcr, median_zcr, std_zcr, kurtosis_zcr, skew_zcr, mean_rms, median_rms, std_rms, kurtosis_rms, skew_rms]


data = pd.read_csv('dataset.csv')

x = data.drop(columns=['Label'], axis =1)
y = data['Label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

# data hasil normalisasi zscorescaler ----------------------------------------------------
with open("zscore_scaler.pkl", 'rb') as file:
    zscore_scaler = pickle.load(file)
zscore_training = zscore_scaler.transform(x_train) #implementasi
zscore_testing = zscore_scaler.transform(x_test)


# Memuat pipeline dari file pickle
with open("pipelinee.pkl", "rb") as file:
    pipeline = pickle.load(file)

# Memisahkan estimator PCA dan KNeighborsClassifier
pca_estimator = pipeline.named_steps["pca"] 
knn_estimator = pipeline.named_steps["knn"]

# Fitting PCA pada data pelatihan
# pca_estimator.fit(zscore_training)

x_train_pca1 = pca_estimator.fit_transform(zscore_training)
x_test_pca1 = pca_estimator.fit_transform(zscore_testing)

knn_estimator.fit(x_train_pca1, y_train)

prediksi_1 = knn_estimator.predict(x_test_pca1)

uploaded_file = st.file_uploader("Pilih file audio...", type=["wav", "mp3"])


if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Cek Prediksi"):
        # Simpan file audio yang diunggah
        audio_path = "audio_diunggah.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Hitung statistik untuk file audio yang diunggah
        statistik = calculate_statistics(audio_path)

        results = []
        result = {
            'Mean': statistik[0],
            'Median': statistik[1],
            'Mode': statistik[2],
            'Max': statistik[3],
            'Min': statistik[4],
            'Std': statistik[5],
            'Skewness': statistik[6],
            'Kurtosis': statistik[7],
            'Q1': statistik[8],
            'Q3': statistik[9],
            'IQR': statistik[10],
            'ZCR_mean': statistik[11],
            'ZCR_median': statistik[12],
            'ZCR_std': statistik[13],

            'ZCR_skewness': statistik[14],
            'ZCR_kurtosis': statistik[15],
            'RMSE_mean': statistik[16],
            'RMSE_median': statistik[17],
            'RMSE_std': statistik[18],
            'RMSE_skewness': statistik[19],
            'RMSE_kurtosis': statistik[20],
        }
        results.append(result)
        df = pd.DataFrame(results)
        st.write(df)

        normalized_features1 = zscore_scaler.transform(df) #data df di normalisasikan
        x_pca1 = pca_estimator.transform(normalized_features1) #data df di reduksi
        prediksi_hasil1 = knn_estimator.predict(x_pca1) #data df di prediksi

        # Menampilkan hasil prediksi
        st.write("Emosi Terdeteksi:", prediksi_hasil1)