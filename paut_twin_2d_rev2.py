import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CONFIGURATION
st.set_page_config(page_title="PAUT 2D Matrix Twin", layout="wide")

# 2. TRADUCTIONS
lang = st.sidebar.selectbox("Langue / Language", ["Français", "English"])
T = {
    "Français": {
        "title": "🛡️ Jumeau Numérique : Sonde Matricielle 2D (Shear Wave)",
        "config": "Configuration de la Sonde",
        "nx": "Nb éléments X", "ny": "Nb éléments Y",
        "px": "Pitch X (mm)", "py": "Pitch Y (mm)",
        "control": "Pilotage du Faisceau (Interférence Constructive)",
        "sector": "Angle Sectoriel (θ)", "skew": "Angle de Skew (φ)",
        "heatmap": "Tenseur de Retards (ns)", "viz3d": "Visualisation du Faisceau (Volume d'Énergie)",
        "info": "Analyse du faisceau", "warn": "⚠️ Attention : Pitch > λ/2 !"
    },
    "English": {
        "title": "🛡️ Digital Twin: 2D Matrix Array (Shear Wave)",
        "config": "Probe Configuration",
        "nx": "Nb elements X", "ny": "Nb elements Y",
        "px": "X Pitch (mm)", "py": "Y Pitch (mm)",
        "control": "Beam Steering (Constructive Interference)",
        "sector": "Sectoral Angle (θ)", "skew": "Skew Angle (φ)",
        "heatmap": "Delay Map (ns)", "viz3d": "Beam Visualization (Energy Volume)",
        "info": "Beam Analysis", "warn": "⚠️ Warning: Pitch > λ/2 !"
    }
}

# 3. SIDEBAR
st.sidebar.header(T[lang]["config"])
nx = st.sidebar.number_input(T[lang]["nx"], 1, 16, 8)
ny = st.sidebar.number_input(T[lang]["ny"], 1, 16, 8)
pitch_x = st.sidebar.slider(T[lang]["px"], 0.1, 1.0, 0.32)
pitch_y = st.sidebar.slider(T[lang]["py"], 0.1, 1.0, 0.32)

v_shear = 3240.0
freq = 5e6
wavelength = (v_shear / freq) * 1000

# 4. CALCUL (CACHE)
@st.cache_data
def compute_laws(nx, ny, px, py, v):
    s_range = np.arange(35, 71, 1)
    k_range = np.arange(-30, 31, 1)
    x = (np.arange(nx) - (nx - 1) / 2) * (px / 1000)
    y = (np.arange(ny) - (ny - 1) / 2) * (py / 1000)
    X, Y = np.meshgrid(x, y)
    X_f, Y_f = X.flatten(), Y.flatten()
    laws = np.zeros((len(s_range), len(k_range), nx * ny))
    for i, s in enumerate(s_range):
        for j, k in enumerate(k_range):
            th, ph = np.radians(s), np.radians(k)
            t = (X_f * np.sin(th) * np.cos(ph) + Y_f * np.sin(th) * np.sin(ph)) / v
            laws[i, j, :] = (t - np.min(t)) * 1e9
    return laws

laws_matrix = compute_laws(nx, ny, pitch_x, pitch_y, v_shear)

# 5. AFFICHAGE
st.title(T[lang]["title"])
st.subheader(T[lang]["control"])

c1, c2 = st.columns(2)
with c1:
    val_s = st.slider(T[lang]["sector"], 35, 70, 55)
with c2:
    val_k = st.slider(T[lang]["skew"], -30, 30, 0)

idx_s, idx_k = val_s - 35, val_k + 30
current_law = laws_matrix[idx_s, idx_k, :].reshape(ny, nx)

col_map, col_viz = st.columns(2)
with col_map:
    st.write(f"**{T[lang]['heatmap']}**")
    fig, ax = plt.subplots()
    sns.heatmap(current_law, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar=False)
    st.pyplot(fig)

with col_viz:
    st.write(f"**{T[lang]['viz3d']}**")
    fig3d = plt.figure(figsize=(7,7))
    ax3d = fig3d.add_subplot(111, projection='3d')
    depth = -40 
    th_r, ph_r = np.radians(val_s), np.radians(val_k)
    z_ax = np.linspace(0, depth, 15)
    x_ax = -z_ax * np.tan(th_r) * np.cos(ph_r)
    y_ax = -z_ax * np.tan(th_r) * np.sin(ph_r)
    for i in range(len(z_ax)):
        r = 2.0 + abs(z_ax[i]*0.1)
        phi = np.linspace(0, 2*np.pi, 12)
        ax3d.plot(x_ax[i] + r*np.cos(phi), y_ax[i] + r*np.sin(phi), z_ax[i], color='red', alpha=0.3)
    ax3d.plot(x_ax, y_ax, z_ax, color='red', lw=3)
    ax3d.set_xlim([-30, 30]); ax3d.set_ylim([-30, 30]); ax3d.set_zlim([-40, 0])
    ax3d.view_init(elev=25, azim=45)
    st.pyplot(fig3d)

if pitch_x > wavelength/2 or pitch_y > wavelength/2:
    st.sidebar.warning(T[lang]["warn"])