"""
app.py  —  Valuador de Terrenos 2025
Carga: preprocessor.pkl + model.pkl + model_info.json
"""

import json, joblib, numpy as np, pandas as pd, streamlit as st

# ── Configuración ─────────────────────────────────────────────
st.set_page_config(
    page_title="Valuador de Terrenos 2025",
    page_icon="🏗️",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.hero {
    background: linear-gradient(135deg, #0a2342 0%, #1a4a7a 60%, #0d6e6e 100%);
    border-radius: 16px; padding: 36px 32px 28px 32px;
    margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.04); border-radius: 50%;
}
.hero h1 { color: white !important; font-size: 2rem !important; margin: 0 0 6px 0 !important; }
.hero .subtitle { color: rgba(255,255,255,0.7); font-size: 0.92rem; margin: 0; }
.badge {
    display: inline-block; background: rgba(255,255,255,0.12); color: #7dd3c8;
    border: 1px solid rgba(125,211,200,0.3); border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem; font-weight: 500;
    margin-bottom: 14px; letter-spacing: 0.5px; text-transform: uppercase;
}
.section-title {
    font-family: 'Syne', sans-serif; font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.5px; color: #888;
    margin: 24px 0 12px 0; border-bottom: 1px solid #eee; padding-bottom: 6px;
}
.result-card {
    background: linear-gradient(135deg, #0a2342 0%, #1a4a7a 100%);
    border-radius: 16px; padding: 32px; text-align: center;
    margin: 24px 0 8px 0; border: 1px solid rgba(125,211,200,0.2);
    box-shadow: 0 8px 32px rgba(10,35,66,0.25);
}
.result-card .label { color: rgba(255,255,255,0.65); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 8px 0; }
.result-card .value { color: #7dd3c8; font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; margin: 0; line-height: 1.1; }
.result-card .range { color: rgba(255,255,255,0.5); font-size: 0.85rem; margin: 10px 0 0 0; }
.mini-metrics { display: flex; gap: 10px; justify-content: center; margin-top: 18px; flex-wrap: wrap; }
.mini-m { background: rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 16px; text-align: center; }
.mini-m .mk { color: rgba(255,255,255,0.5); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; }
.mini-m .mv { color: white; font-size: 1rem; font-weight: 600; }
.disclaimer {
    background: #fffbeb; border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    font-size: 0.82rem; color: #78350f; margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Carga de artefactos ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('preprocessor.pkl')
    model        = joblib.load('model.pkl')
    with open('model_info.json') as f:
        info = json.load(f)
    return preprocessor, model, info

try:
    preprocessor, model, info = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"❌ Archivo no encontrado: `{e.filename}`\n\n"
        "Asegúrate de que `preprocessor.pkl`, `model.pkl` y `model_info.json` "
        "estén en la raíz del repositorio."
    )
    st.stop()

ALL_FEATURES = info['all_features']
mejor_modelo = info['mejor_modelo']
metricas     = info['metricas']


# ── HERO ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="badge">🏗️ ML · Mercado Inmobiliario Perú</div>
    <h1>Valuador de Terrenos 2025</h1>
    <p class="subtitle">
        Modelo: <strong style="color:#7dd3c8">{mejor_modelo}</strong>
        &nbsp;·&nbsp; R² = {metricas['R2']:.4f}
        &nbsp;·&nbsp; MAPE = {metricas['MAPE']:.2f}%
        &nbsp;·&nbsp; MAE = S/ {metricas['MAE']:,.0f}
    </p>
</div>
""", unsafe_allow_html=True)


# ── FORMULARIO ────────────────────────────────────────────────
input_vals = {}

# ── Áreas principales ─────────────────────────────────────────
st.markdown('<div class="section-title">📐 Áreas del predio</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    input_vals['area_terreno'] = st.number_input(
        "Área terreno (m²)", min_value=0.0, value=120.0, step=5.0)
with col2:
    input_vals['area_construida'] = st.number_input(
        "Área construida (m²)", min_value=0.0, value=80.0, step=5.0)
with col3:
    input_vals['area_total_construida'] = st.number_input(
        "Área total construida (m²)", min_value=0.0, value=80.0, step=5.0)

col4, col5, col6 = st.columns(3)
with col4:
    input_vals['area_comun_terreno'] = st.number_input(
        "Área común terreno (m²)", min_value=0.0, value=0.0, step=1.0)
with col5:
    input_vals['area_comun_construida'] = st.number_input(
        "Área común construida (m²)", min_value=0.0, value=0.0, step=1.0)
with col6:
    # Calculado automáticamente
    area_comun_total = input_vals['area_comun_terreno'] + input_vals['area_comun_construida']
    input_vals['area_comun_total'] = area_comun_total
    st.metric("Área común total (m²)", f"{area_comun_total:.1f}")

# Features derivadas (calculadas, no ingresadas)
input_vals['area_suma'] = input_vals['area_terreno'] + input_vals['area_total_construida']
input_vals['ratio_construido'] = (
    input_vals['area_total_construida'] / input_vals['area_terreno']
    if input_vals['area_terreno'] > 0 else 0
)


# ── Edificación ───────────────────────────────────────────────
st.markdown('<div class="section-title">🏢 Edificación</div>', unsafe_allow_html=True)
col7, col8, col9 = st.columns(3)
with col7:
    input_vals['pisos'] = st.number_input(
        "Pisos", min_value=1, max_value=50, value=3, step=1)
with col8:
    anio_construccion = st.number_input(
        "Año construcción", min_value=1800, max_value=2025, value=2000, step=1)
    input_vals['edad_predio'] = 2025 - anio_construccion
with col9:
    input_vals['material_predio'] = st.selectbox(
        "Material del predio", [
            'Concreto',
            'Ladrillo',
            'Madera u otros',
            'Adobe',
            'No declarado',
        ]
    )


# ── Propiedad ─────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Propiedad</div>', unsafe_allow_html=True)
col10, col11, col12 = st.columns(3)
with col10:
    input_vals['pct_propiedad'] = st.number_input(
        "% propiedad", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
with col11:
    anio_adquisicion = st.number_input(
        "Año adquisición", min_value=1900, max_value=2025, value=2010, step=1)
    input_vals['antiguedad_propiedad'] = 2025 - anio_adquisicion
with col12:
    input_vals['tipo_propietario'] = st.selectbox(
        "Tipo propietario", [
            'Propietario Único',
            'Condómino',
            'Poseedor',
            'Responsable',
            'Concesionario',
        ]
    )


# ── Uso ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">🏷️ Uso del predio</div>', unsafe_allow_html=True)
input_vals['uso_macro'] = st.selectbox(
    "Uso del predio", [
        'Vivienda',
        'Comercial',
        'Servicios',
        'Industrial',
        'Educación',
        'Salud',
        'Recreación',
        'Sin construir',
        'Otros',
    ]
)


# ── PREDICCIÓN ────────────────────────────────────────────────
st.markdown("")
if st.button("💰  Calcular valor del terreno", use_container_width=True, type="primary"):

    X_input  = pd.DataFrame([{k: input_vals[k] for k in ALL_FEATURES}])
    X_pp     = preprocessor.transform(X_input)
    log_pred = model.predict(X_pp)[0]
    valor    = np.expm1(log_pred)

    margen = metricas['MAPE'] / 100
    low    = valor * (1 - margen)
    high   = valor * (1 + margen)
    val_m2 = valor / input_vals['area_terreno'] if input_vals['area_terreno'] > 0 else 0

    st.markdown(f"""
    <div class="result-card">
        <p class="label">Valor estimado del terreno</p>
        <p class="value">S/ {valor:,.0f}</p>
        <p class="range">Rango estimado: S/ {low:,.0f} — S/ {high:,.0f}</p>
        <div class="mini-metrics">
            <div class="mini-m">
                <div class="mk">Valor / m²</div>
                <div class="mv">S/ {val_m2:,.0f}</div>
            </div>
            <div class="mini-m">
                <div class="mk">R²</div>
                <div class="mv">{metricas['R2']:.4f}</div>
            </div>
            <div class="mini-m">
                <div class="mk">MAPE</div>
                <div class="mv">{metricas['MAPE']:.2f}%</div>
            </div>
            <div class="mini-m">
                <div class="mk">MAE</div>
                <div class="mv">S/ {metricas['MAE']:,.0f}</div>
            </div>
        </div>
    </div>
    <div class="disclaimer">
        ⚠️ Esta estimación es referencial y no reemplaza una tasación profesional registrada.
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.caption("Modelo entrenado con datos del mercado inmobiliario peruano · 2025")
