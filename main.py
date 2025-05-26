import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
from dotenv import load_dotenv
import os

# Setup
st.set_page_config(page_title="Analiză voturi prezidențiale 2025", layout="wide")
st.title("🗳️ Analiză creșteri și anomalii voturi - Tur 1 vs Tur 2")

# Linkuri descarcare CSV
st.markdown("""
### 📥 Descarcă fișierele CSV pentru Turul 1 și Turul 2:
- [Descarcă Turul 1 (Google Drive)](https://drive.google.com/uc?export=download&id=1LiB1QDutQO-OCK-qMabMKwsklDt1gicp)
- [Descarcă Turul 2 (Google Drive)](https://drive.google.com/uc?export=download&id=1rRlfKm2u3N6TckYeDteO1fPAlgoBH9bg)
- [Date oficiale prezenta.roaep.ro](https://prezenta.roaep.ro/prezidentiale18052025/pv/results)
""")

# Sidebar Help / Explicații
with st.sidebar.expander("❓ Help / Explicații rapoarte"):
    st.markdown("""
    ### Cum să interpretezi raportul de voturi între turul 1 și turul 2

    - **Delta GS / Delta ND**: diferența numerică de voturi obținute de candidați între turul 2 și turul 1.  
      Exemplu: dacă „Delta GS” = 45, înseamnă că George Simion a primit cu 45 de voturi mai mult în turul 2 față de turul 1.

    - **Raport GS / Raport ND**: raportul dintre voturile din turul 2 și cele din turul 1.  
      Exemplu: dacă „Raport GS” = 1.45, înseamnă o creștere cu 45% față de turul 1 (1.00 = nicio schimbare, 0.8 = scădere cu 20%).

    - **Z-score**: măsoară cât de mult diferă o valoare de media întregului set de date, în unități de deviație standard.  
      Un Z-score mare (ex: > 3) indică o schimbare neobișnuită, posibil suspectă.

    - **Anomalie IF (Isolation Forest)**: algoritm de detecție automată a valorilor „atipice” care nu se încadrează în modelul normal.  
      Dacă ești marcat ca anomalie, înseamnă că schimbarea voturilor e neobișnuită comparativ cu alte secții/zone.

    ---
    Poți selecta tipurile de anomalii în sidebar pentru a filtra tabelul și graficul.
    """)

# Upload CSVs
uploaded_file1 = st.file_uploader("Încarcă fișier CSV Tur 1", type=["csv"])
uploaded_file2 = st.file_uploader("Încarcă fișier CSV Tur 2", type=["csv"])

if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    rename_map = {
        'precinct_county_name': 'Județ',
        'precinct_county_nce': 'Cod Județ',
        'precinct_name': 'Secție',
        'precinct_nr': 'cod_secție',
        'uat_name': 'Localitate',
        'GEORGE-NICOLAE SIMION-voturi': 'GS_voturi',
        'NICUȘOR-DANIEL DAN-voturi': 'ND_voturi',
    }

    cols_needed = list(rename_map.keys())
    df1 = df1[cols_needed].rename(columns=rename_map)
    df2 = df2[cols_needed].rename(columns=rename_map)

    for df in [df1, df2]:
        df['GS_voturi'] = pd.to_numeric(df['GS_voturi'], errors='coerce')
        df['ND_voturi'] = pd.to_numeric(df['ND_voturi'], errors='coerce')

    df = pd.merge(df1, df2, on=['cod_secție', 'Județ', 'Cod Județ', 'Localitate', 'Secție'], suffixes=('_t1', '_t2'))
    df.dropna(subset=['GS_voturi_t1', 'GS_voturi_t2', 'ND_voturi_t1', 'ND_voturi_t2'], inplace=True)

    # Calcul diferențe și rapoarte
    df['Delta GS'] = df['GS_voturi_t2'] - df['GS_voturi_t1']
    df['Delta ND'] = df['ND_voturi_t2'] - df['ND_voturi_t1']
    df['Ratio GS'] = df['GS_voturi_t2'] / df['GS_voturi_t1'].replace(0, np.nan)
    df['Ratio ND'] = df['ND_voturi_t2'] / df['ND_voturi_t1'].replace(0, np.nan)

    # Anomalii Isolation Forest
    contamination = 0.05
    for col in ['Delta GS', 'Delta ND', 'Ratio GS', 'Ratio ND']:
        df_valid = df[[col]].dropna()
        clf = IsolationForest(contamination=contamination, random_state=42)
        df.loc[df_valid.index, f'Anomalie IF {col}'] = clf.fit_predict(df_valid)
        df[f'Anomalie IF {col}'] = df[f'Anomalie IF {col}'].map({1: False, -1: True})

    # Anomalii Z-score
    z_threshold = 3
    for col in ['Delta GS', 'Delta ND']:
        mean = df[col].mean()
        std = df[col].std()
        df[f'Zscore {col}'] = (df[col] - mean) / std
        df[f'Anomalie Z {col}'] = df[f'Zscore {col}'].abs() > z_threshold

    # Agregare
    agregare = st.selectbox("Agregare date după:", options=['Secție', 'Localitate', 'Județ'])

    if agregare == 'Secție':
        df_agg = df.copy()
        group_cols = ['cod_secție', 'Secție', 'Localitate', 'Județ']
    elif agregare == 'Localitate':
        df_agg = df.groupby(['Localitate', 'Județ'], as_index=False).agg({
            'Delta GS': 'sum', 'Delta ND': 'sum',
            'Ratio GS': 'mean', 'Ratio ND': 'mean',
            'Anomalie IF Delta GS': 'any', 'Anomalie IF Delta ND': 'any',
            'Anomalie IF Ratio GS': 'any', 'Anomalie IF Ratio ND': 'any',
            'Anomalie Z Delta GS': 'any', 'Anomalie Z Delta ND': 'any'
        })
        group_cols = ['Localitate', 'Județ']
    else:
        df_agg = df.groupby(['Județ'], as_index=False).agg({
            'Delta GS': 'sum', 'Delta ND': 'sum',
            'Ratio GS': 'mean', 'Ratio ND': 'mean',
            'Anomalie IF Delta GS': 'any', 'Anomalie IF Delta ND': 'any',
            'Anomalie IF Ratio GS': 'any', 'Anomalie IF Ratio ND': 'any',
            'Anomalie Z Delta GS': 'any', 'Anomalie Z Delta ND': 'any'
        })
        group_cols = ['Județ']

    # Filtru anomalii
    st.sidebar.header("Filtrare secții/zone cu anomalii")
    anomaly_filter_options = st.sidebar.multiselect(
        "Selectează tipuri anomalii de afișat:",
        options=[col for col in df_agg.columns if 'Anomalie' in col],
        default=[]
    )

    if anomaly_filter_options:
        mask = np.zeros(len(df_agg), dtype=bool)
        for opt in anomaly_filter_options:
            if opt in df_agg.columns:
                mask |= df_agg[opt].astype(bool)
        df_filtered = df_agg[mask]
    else:
        df_filtered = df_agg.copy()

    st.subheader(f"Tabel - {agregare} {'cu anomalii filtrate' if anomaly_filter_options else '(toate)'}")
    st.dataframe(df_filtered)

    # Grafic creștere
    st.subheader(f"Grafic creșteri (Raport Tur 2 / Tur 1) pe {agregare}")
    df_long = df_filtered.melt(id_vars=group_cols,
                               value_vars=['Ratio GS', 'Ratio ND'],
                               var_name='Candidat',
                               value_name='Raport creștere')
    color_map = {'Ratio GS': 'blue', 'Ratio ND': 'red'}
    df_long['x_label'] = df_long[group_cols].astype(str).agg(" - ".join, axis=1)

    fig = px.scatter(
        df_long,
        x='x_label',
        y='Raport creștere',
        color='Candidat',
        color_discrete_map=color_map,
        title=f"Raport creștere voturi Tur2 / Tur1 pe {agregare}",
        labels={'x_label': agregare, 'Raport creștere': 'Raport creștere'},
        hover_data=group_cols
    )
    fig.update_layout(xaxis_title=agregare, xaxis_showticklabels=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
