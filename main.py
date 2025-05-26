import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

# Setup
st.set_page_config(page_title="Analiză voturi prezidențiale 2025", layout="wide")
st.title("🗳️ Analiză creșteri și anomalii voturi - Tur 1 vs Tur 2")

# Linkuri descărcare CSV
st.markdown("""
### 📥 Descarcă fișierele CSV pentru Turul 1 și Turul 2:
- [Descarcă Turul 1 (Google Drive)](https://drive.google.com/uc?export=download&id=1LiB1QDutQO-OCK-qMabMKwsklDt1gicp)
- [Descarcă Turul 2 (Google Drive)](https://drive.google.com/uc?export=download&id=1rRlfKm2u3N6TckYeDteO1fPAlgoBH9bg)
- [Date oficiale prezenta.roaep.ro](https://prezenta.roaep.ro/prezidentiale18052025/pv/results)
""")

# Sidebar Help cu explicații tehnice detaliate
with st.sidebar.expander("❓ Help / Explicații rapoarte"):
    st.markdown("""
    ### Cum să interpretezi raportul de voturi între turul 1 și turul 2
    - **Delta GS / Delta ND**: Diferența absolută de voturi între tururi.
      - Valori pozitive indică creșteri, valori negative pot sugera scăderi sau anomalii.
    - **Raport GS / ND**: Raportul voturi Tur 2 / Tur 1.
      - Valori peste 1 înseamnă creștere relativă, valori foarte mari pot indica mobilizare neobișnuită.
    - **Delta Participare**: Diferența ratei de participare (prezență/alegători înscriși) între tururi.
    - **Voturi nule %**: Procentul de voturi nule din totalul voturilor exprimate.
    - **Suplimentare %**: Procentul voturilor din lista suplimentară din prezența totală.
    - **Z-score**: Măsoară abaterile față de media tuturor secțiilor, în unități de deviație standard.
      - Z-score mare (ex. >3) semnalează secții cu valori neobișnuite statistic.
    - **Anomalie IF**: Detectare automată cu Isolation Forest, un algoritm ML care identifică valori atipice.
    """)

# Funcție pentru detectarea anomaliilor
@st.cache_data
def detect_anomalies(df, columns, contamination=0.05):
    clf = IsolationForest(contamination=contamination, random_state=42)
    for col in columns:
        df_valid = df[[col]].dropna()
        df.loc[df_valid.index, f'Anomalie IF {col}'] = clf.fit_predict(df_valid) == -1
    return df

# Upload CSVs
uploaded_file1 = st.file_uploader("Încarcă fișier CSV Tur 1", type=["csv"])
uploaded_file2 = st.file_uploader("Încarcă fișier CSV Tur 2", type=["csv"])

if uploaded_file1 and uploaded_file2:
    # Încărcare și curățare date
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Harta redenumire coloane
    rename_map = {
        'precinct_county_name': 'Județ',
        'precinct_county_nce': 'Cod Județ',
        'precinct_name': 'Secție',
        'precinct_nr': 'cod_secție',
        'uat_name': 'Localitate',
        'GEORGE-NICOLAE SIMION-voturi': 'GS_voturi',
        'NICUȘOR-DANIEL DAN-voturi': 'ND_voturi',
        'a': 'alegatori_inscrisi',
        'b': 'prezenta',
        'b1': 'prezenta_lista_permanenta',
        'b2': 'prezenta_lista_suplimentara',
        'c': 'voturi_valabile',
        'd': 'voturi_nule'
    }

    df1 = df1.rename(columns={k: v for k, v in rename_map.items() if k in df1.columns})
    df2 = df2.rename(columns={k: v for k, v in rename_map.items() if k in df2.columns})

    # Coloane esențiale
    cols_t1 = ['cod_secție', 'Județ', 'Cod Județ', 'Localitate', 'Secție', 'alegatori_inscrisi', 'prezenta',
               'prezenta_lista_permanenta', 'prezenta_lista_suplimentara', 'voturi_valabile', 'voturi_nule',
               'GS_voturi', 'ND_voturi']
    cols_t2 = cols_t1

    # Verificare coloane lipsă
    missing_cols_t1 = [col for col in cols_t1 if col not in df1.columns]
    missing_cols_t2 = [col for col in cols_t2 if col not in df2.columns]
    if missing_cols_t1:
        st.warning(f"Fișier Tur 1 lipsește coloanele: {missing_cols_t1}. Analiza va continua fără acestea.")
    if missing_cols_t2:
        st.warning(f"Fișier Tur 2 lipsește coloanele: {missing_cols_t2}. Analiza va continua fără acestea.")

    # Selectăm doar coloanele disponibile
    available_cols_t1 = [col for col in cols_t1 if col in df1.columns]
    available_cols_t2 = [col for col in cols_t2 if col in df2.columns]
    df1 = df1[available_cols_t1]
    df2 = df2[available_cols_t2]

    # Conversie la numeric
    numeric_cols = ['alegatori_inscrisi', 'prezenta', 'prezenta_lista_permanenta', 'prezenta_lista_suplimentara',
                    'voturi_valabile', 'voturi_nule', 'GS_voturi', 'ND_voturi']
    for df in [df1, df2]:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Validări consistență date
    for df, tur in [(df1, 'Tur 1'), (df2, 'Tur 2')]:
        if 'prezenta' in df.columns and 'prezenta_lista_permanenta' in df.columns and 'prezenta_lista_suplimentara' in df.columns:
            if (df['prezenta'] < df['prezenta_lista_permanenta'] + df['prezenta_lista_suplimentara']).any():
                st.warning(f"Inconsistență în {tur}: Prezența totală este mai mică decât suma listei permanente și suplimentare.")
        if 'voturi_valabile' in df.columns and 'voturi_nule' in df.columns and 'prezenta' in df.columns:
            if (df['voturi_valabile'] + df['voturi_nule'] > df['prezenta']).any():
                st.warning(f"Inconsistență în {tur}: Voturile valabile + nule depășesc prezența.")

    # Merge date
    df = pd.merge(df1, df2, on=['cod_secție', 'Județ', 'Cod Județ', 'Localitate', 'Secție'], suffixes=('_t1', '_t2'))
    df.dropna(subset=['GS_voturi_t1', 'GS_voturi_t2', 'ND_voturi_t1', 'ND_voturi_t2'], inplace=True)

    # Calcule suplimentare
    df['Delta GS'] = df['GS_voturi_t2'] - df['GS_voturi_t1']
    df['Delta ND'] = df['ND_voturi_t2'] - df['ND_voturi_t1']
    df['Ratio GS'] = df['GS_voturi_t2'] / df['GS_voturi_t1'].replace(0, np.nan)
    df['Ratio ND'] = df['ND_voturi_t2'] / df['ND_voturi_t1'].replace(0, np.nan)
    if 'alegatori_inscrisi_t1' in df.columns and 'prezenta_t1' in df.columns:
        df['Participare_t1'] = df['prezenta_t1'] / df['alegatori_inscrisi_t1'].replace(0, np.nan)
        df['Participare_t2'] = df['prezenta_t2'] / df['alegatori_inscrisi_t2'].replace(0, np.nan)
        df['Delta Participare'] = df['Participare_t2'] - df['Participare_t1']
    if 'voturi_nule_t2' in df.columns and 'voturi_valabile_t2' in df.columns:
        df['Voturi nule %'] = df['voturi_nule_t2'] / (df['voturi_valabile_t2'] + df['voturi_nule_t2']).replace(0, np.nan) * 100
    if 'prezenta_lista_suplimentara_t2' in df.columns and 'prezenta_t2' in df.columns:
        df['Suplimentare %'] = df['prezenta_lista_suplimentara_t2'] / df['prezenta_t2'].replace(0, np.nan) * 100

    # Detectare anomalii
    contamination = st.sidebar.slider("Sensibilitate anomalii (Isolation Forest)", 0.01, 0.1, 0.05, 0.01)
    anomaly_cols = ['Delta GS', 'Delta ND', 'Ratio GS', 'Ratio ND']
    if 'Delta Participare' in df.columns:
        anomaly_cols.append('Delta Participare')
    if 'Voturi nule %' in df.columns:
        anomaly_cols.append('Voturi nule %')
    if 'Suplimentare %' in df.columns:
        anomaly_cols.append('Suplimentare %')
    df = detect_anomalies(df, anomaly_cols, contamination)

    # Calcule Z-score
    z_threshold = st.sidebar.slider("Prag Z-score", 1.0, 5.0, 3.0, 0.1)
    for col in ['Delta GS', 'Delta ND'] + (['Delta Participare'] if 'Delta Participare' in df.columns else []):
        mean = df[col].mean()
        std = df[col].std()
        df[f'Zscore {col}'] = (df[col] - mean) / std
        df[f'Anomalie Z {col}'] = df[f'Zscore {col}'].abs() > z_threshold

    # Statistici sumare
    st.subheader("Statistici sumare")
    col1, col2, col3 = st.columns(3)
    col1.metric("Media Delta GS", f"{df['Delta GS'].mean():.2f}")
    col2.metric("Media Delta ND", f"{df['Delta ND'].mean():.2f}")
    if 'Delta Participare' in df.columns:
        col3.metric("Media Delta Participare", f"{df['Delta Participare'].mean():.4f}")
    col1.metric("Procent anomalii Delta GS", f"{(df['Anomalie IF Delta GS'].mean() * 100):.1f}%")
    if 'Voturi nule %' in df.columns:
        col2.metric("Media voturi nule %", f"{df['Voturi nule %'].mean():.1f}%")
    if 'Suplimentare %' in df.columns:
        col3.metric("Media suplimentare %", f"{df['Suplimentare %'].mean():.1f}%")

    # Alegere nivel agregare
    agregare = st.selectbox("Agregare date după:", options=['Secție', 'Localitate', 'Județ'])
    if agregare == 'Secție':
        df_agg = df.copy()
        group_cols = ['cod_secție', 'Secție', 'Localitate', 'Județ']
    else:
        group = ['Localitate', 'Județ'] if agregare == 'Localitate' else ['Județ']
        anomaly_cols = [col for col in df.columns if col.startswith('Anomalie')]
        agg_dict = {
            'Delta GS': 'sum',
            'Delta ND': 'sum',
            'Ratio GS': 'mean',
            'Ratio ND': 'mean'
        }
        if 'Delta Participare' in df.columns:
            agg_dict['Delta Participare'] = 'mean'
        if 'Voturi nule %' in df.columns:
            agg_dict['Voturi nule %'] = 'mean'
        if 'Suplimentare %' in df.columns:
            agg_dict['Suplimentare %'] = 'mean'
        for col in anomaly_cols:
            df[col] = df[col].astype(bool)
            df[f'%{col}'] = df[col].astype(int)
            agg_dict[f'%{col}'] = 'mean'
        df_agg = df.groupby(group, as_index=False).agg(agg_dict)
        for col in anomaly_cols:
            df_agg[f'%{col}'] = (df_agg[f'%{col}'] * 100).round(1)
        group_cols = group

    # Filtru anomalii
    st.sidebar.header("Filtrare secții/zone cu anomalii")
    anomaly_filter_options = st.sidebar.multiselect(
        "Selectează tipuri anomalii de afișat:",
        options=[col for col in df_agg.columns if 'Anomalie' in col or '%Anomalie' in col],
        default=[]
    )

    if anomaly_filter_options:
        mask = np.zeros(len(df_agg), dtype=bool)
        for opt in anomaly_filter_options:
            if opt in df_agg.columns:
                mask |= df_agg[opt] > 0 if '%' in opt else df_agg[opt].astype(bool)
        df_filtered = df_agg[mask]
    else:
        df_filtered = df_agg.copy()

    # Afișare tabel
    st.subheader(f"Tabel - {agregare} {'cu anomalii filtrate' if anomaly_filter_options else '(toate)'}")
    st.dataframe(df_filtered)

    # Descărcare CSV
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Descarcă datele filtrate (CSV)", data=csv, file_name="date_filtrate.csv", mime='text/csv')

    # Grafic Plotly - Raport voturi
    st.subheader(f"Grafic creșteri (Raport Tur 2 / Tur 1) pe {agregare}")
    df_long = df_filtered.melt(id_vars=group_cols,
                               value_vars=['Ratio GS', 'Ratio ND'],
                               var_name='Candidat',
                               value_name='Raport creștere')
    df_long['x_label'] = df_long[group_cols].astype(str).agg(" - ".join, axis=1)
    color_map = {'Ratio GS': 'blue', 'Ratio ND': 'red'}
    fig = px.scatter(
        df_long,
        x='x_label',
        y='Raport creștere',
        color='Candidat',
        color_discrete_map=color_map,
        title=f"Raport creștere voturi Tur2 / Tur1 pe {agregare}",
        labels={'x_label': agregare, 'Raport creștere': 'Raport creștere'},
        hover_data=group_cols,
        log_y=True
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(xaxis_title=agregare, xaxis_showticklabels=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Grafic suplimentar - Participare și voturi nule
    show_extra_plot = st.sidebar.checkbox("Afișează grafic participare și voturi nule", value=False)
    if show_extra_plot and 'Delta Participare' in df_filtered.columns and 'Voturi nule %' in df_filtered.columns:
        st.subheader(f"Grafic participare și voturi nule pe {agregare}")
        df_long_extra = df_filtered.melt(id_vars=group_cols,
                                         value_vars=['Delta Participare', 'Voturi nule %'],
                                         var_name='Metrică',
                                         value_name='Valoare')
        df_long_extra['x_label'] = df_long_extra[group_cols].astype(str).agg(" - ".join, axis=1)
        color_map_extra = {'Delta Participare': 'green', 'Voturi nule %': 'purple'}
        fig_extra = px.scatter(
            df_long_extra,
            x='x_label',
            y='Valoare',
            color='Metrică',
            color_discrete_map=color_map_extra,
            title=f"Participare și voturi nule pe {agregare}",
            labels={'x_label': agregare, 'Valoare': 'Valoare'},
            hover_data=group_cols
        )
        fig_extra.update_traces(marker=dict(size=4))
        fig_extra.update_layout(xaxis_title=agregare, xaxis_showticklabels=False, height=600)
        st.plotly_chart(fig_extra, use_container_width=True)
