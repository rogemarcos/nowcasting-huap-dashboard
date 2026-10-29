import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier

# Configuración de página
st.set_page_config(page_title="Nowcasting Urgencias (Random Forest)", layout="wide")

# 1. Carga y preparación de datos base
@st.cache_data
def load_data():
    df = pd.read_csv('datos_urgencias_santiago_centro_20km_tuned_staff_rn16_extreme_targets.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df_base = load_data()

# 2. Pipeline de Machine Learning (Entrenamiento de Random Forest)
@st.cache_resource
def train_rf_models(df):
    targets = [f"contingency_extreme_t+{i}" for i in range(1, 7)]
    X = df.drop(columns=["timestamp"] + targets)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    models = {}
    for target in targets:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_encoded, df[target])
        models[target] = rf
        
    return models, X_encoded

models, X_encoded = train_rf_models(df_base)

# 3. Lógica de Estado para la navegación temporal
min_idx = 23  
max_idx = len(df_base) - 7

if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 9400

def retroceder_hora():
    if st.session_state.current_idx > min_idx:
        st.session_state.current_idx -= 1

def avanzar_hora():
    if st.session_state.current_idx < max_idx:
        st.session_state.current_idx += 1

# ========================================
# BARRA LATERAL (SIDEBAR)
# ========================================
st.sidebar.title("⏱️ Simulador de Tiempo")
col_btn1, col_btn2 = st.sidebar.columns(2)
col_btn1.button("⏪ -1 Hora", on_click=retroceder_hora, use_container_width=True)
col_btn2.button("+1 Hora ⏩", on_click=avanzar_hora, use_container_width=True)

st.sidebar.slider(
    "Selecciona el índice temporal:",
    min_value=min_idx,
    max_value=max_idx,
    key="current_idx"
)

st.sidebar.divider()

st.sidebar.title("⚙️ Ponderadores del Índice")
st.sidebar.markdown("Personaliza el peso para el Índice de Carga.")
w_arrival = st.sidebar.number_input("Peso: Tasa de Llegadas", value=1.0, step=0.5)
w_waiting = st.sidebar.number_input("Peso: Pacientes en Espera", value=1.0, step=0.5)
w_boarding = st.sidebar.number_input("Peso: Boarding (Espera Cama)", value=2.0, step=0.5)

# ========================================
# PROCESAMIENTO DINÁMICO DE DATOS
# ========================================
df = df_base.copy()
df['load_index'] = (df['arrival_rate'] * w_arrival) + \
                   (df['waiting_count'] * w_waiting) + \
                   (df['boarding_count'] * w_boarding)

mean_load = df['load_index'].mean()
threshold = mean_load + 2 * df['load_index'].std()
mean_arrival = df['arrival_rate'].mean()
mean_wait = df['waiting_count'].mean()
mean_board = df['boarding_count'].mean()

current_idx = st.session_state.current_idx
now_row = df.iloc[current_idx]
now_time = now_row['timestamp']

df_past = df.iloc[current_idx-23 : current_idx+1].copy()

current_features = X_encoded.iloc[[current_idx]]
rf_predictions = [models[f'contingency_extreme_t+{i}'].predict(current_features)[0] for i in range(1, 7)]

st.sidebar.info(f"**Fecha y hora simulada:**\n\n🕒 {now_time.strftime('%A, %d %b %Y %H:%M')}")

# ========================================
# HEADER Y PESTAÑAS (TABS)
# ========================================
st.title("🏥 NOWCASTING SERVICIO DE URGENCIA")
st.markdown(f"**Foco:** Prevención de Saturación Extrema | **Modelo Predictivo:** Random Forest Classifier[cite: 1]")

# Crear las dos pestañas
tab1, tab2 = st.tabs(["📊 Dashboard Operativo", "🧠 Análisis del Modelo (Importancia de Variables)"])

# ========================================
# PESTAÑA 1: DASHBOARD OPERATIVO
# ========================================
with tab1:
    # KPIs
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    col_kpi1.metric("Llegadas/hora", f"{now_row['arrival_rate']:.1f}")
    col_kpi2.metric("Pacientes en Espera", int(now_row['waiting_count']))
    col_kpi3.metric("Boarding (Esperando Cama)", int(now_row['boarding_count']))
    load_delta_str = f"{- (threshold - now_row['load_index']):.1f} del umbral"
    col_kpi4.metric(f"Índice de Carga (Umbral: {threshold:.1f})", f"{now_row['load_index']:.1f}", delta=load_delta_str, delta_color="inverse")
    riesgo_t1 = "🔴 CRÍTICO" if rf_predictions[0] == 1 else "🟢 NORMAL"
    col_kpi5.metric("Predicción RF (Próx. hora)", riesgo_t1)

    st.divider()

    # Gráficos Principales
    col_chart1, col_chart2 = st.columns(2)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    times = pd.to_datetime(df_past['timestamp']).dt.strftime('%d/%m %H:%M')

    with col_chart1:
        st.subheader("📈 Evolución Índice de Carga (Últimas 24 hrs)")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        
        ax1.plot(times, df_past['load_index'], marker='o', markersize=4, color='#2c3e50', linewidth=2.5, label='Carga Observada')
        ax1.axhline(mean_load, color='#8e44ad', linestyle='-', linewidth=2, label='Media Histórica')
        ax1.axhline(mean_load + 3, color='#9b59b6', linestyle=':', linewidth=1.5, alpha=0.7, label='Media + 3 Pts')
        ax1.axhline(mean_load - 3, color='#9b59b6', linestyle=':', linewidth=1.5, alpha=0.7, label='Media - 3 Pts')
        
        ax1.axhspan(mean_load - 3, mean_load + 3, color='lightgray', alpha=0.4, label='Banda ±3 Pts')
        
        ax1.axhline(threshold, color='#e74c3c', linestyle='--', linewidth=2, label=f'Umbral Extremo ({threshold:.1f})')
        ax1.fill_between(times, df_past['load_index'], threshold, where=(df_past['load_index'] >= threshold), color='#e74c3c', alpha=0.3)
        
        ax1.set_ylabel('Carga (Pts)')
        ax1.legend(loc='upper left', fontsize='small')
        plt.xticks(rotation=45, ha='right')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
        st.pyplot(fig1)

    with col_chart2:
        st.subheader("📊 Desglose de Carga Ponderada")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        
        arr_weighted = df_past['arrival_rate'] * w_arrival
        wait_weighted = df_past['waiting_count'] * w_waiting
        board_weighted = df_past['boarding_count'] * w_boarding
        
        ax2.bar(times, arr_weighted, label=f'Llegadas (x{w_arrival})', color='#3498db', alpha=0.8)
        ax2.bar(times, wait_weighted, bottom=arr_weighted, label=f'En Espera (x{w_waiting})', color='#f1c40f', alpha=0.8)
        ax2.bar(times, board_weighted, bottom=arr_weighted + wait_weighted, label=f'Boarding (x{w_boarding})', color='#e74c3c', alpha=0.8)
        ax2.axhline(threshold, color='#c0392b', linestyle='--', linewidth=1.5)
        
        ax2.legend(loc='upper left', fontsize='small')
        plt.xticks(rotation=45, ha='right')
        ax2.xaxis.set_major_locator(plt.MaxNLocator(12))
        st.pyplot(fig2)

    # Gráficos Individuales
    st.divider()
    st.subheader("🔍 Análisis Individual por Componente")
    col_comp1, col_comp2, col_comp3 = st.columns(3)

    def plot_component(df_subset, col_name, title, color_hex, mean_ref):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(times, df_subset[col_name], marker='o', markersize=3, color=color_hex, linewidth=2, label='Observado')
        ax.axhline(mean_ref, color='#7f8c8d', linestyle='-', linewidth=1.5, label='Media Histórica')
        ax.axhline(mean_ref + 3, color='#95a5a6', linestyle=':', linewidth=1.2, label='+ 3 Pts')
        ax.axhline(mean_ref - 3, color='#95a5a6', linestyle=':', linewidth=1.2, label='- 3 Pts')
        ax.axhspan(mean_ref - 3, mean_ref + 3, color='lightgray', alpha=0.4)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', fontsize=7)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))
        return fig

    with col_comp1:
        st.pyplot(plot_component(df_past, 'arrival_rate', 'Tasa de Llegadas', '#3498db', mean_arrival))
    with col_comp2:
        st.pyplot(plot_component(df_past, 'waiting_count', 'Pacientes en Espera', '#f1c40f', mean_wait))
    with col_comp3:
        st.pyplot(plot_component(df_past, 'boarding_count', 'Boarding (Esperando Cama)', '#e74c3c', mean_board))

    # Predicción y Mapa de Calor Histórico
    st.divider()
    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        st.subheader("🔮 Predicción ML - Random Forest (Próximas 6 Horas)")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        
        future_hours = [(now_time + timedelta(hours=i)).strftime('%H:%M') for i in range(1, 7)]
        colors = ['#c0392b' if t == 1 else '#2ecc71' for t in rf_predictions]

        bars = ax3.bar(future_hours, [1]*6, color=colors, edgecolor='black', alpha=0.8)
        for i, (bar, t) in enumerate(zip(bars, rf_predictions)):
            label = "¡SATURACIÓN!" if t == 1 else "Normal"
            ax3.text(bar.get_x() + bar.get_width()/2, 0.5, label, ha='center', va='center', color='white', fontweight='bold', rotation=90)
            ax3.text(bar.get_x() + bar.get_width()/2, 1.05, f"T+{i+1}", ha='center', va='bottom', fontweight='bold')

        ax3.set_ylim(0, 1.2)
        ax3.set_yticks([])
        st.pyplot(fig3)

    with col_chart4:
        st.subheader("🔥 Probabilidad Histórica de Casos Extremos")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        
        hm_data = df_base.groupby(['day_of_week', 'hour_of_day'])['extreme_case'].mean().unstack()
        days_es = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        hm_data.index = days_es
        sns.heatmap(hm_data, cmap="YlOrRd", ax=ax4, cbar_kws={'label': 'Probabilidad'})
        ax4.set_xlabel('Hora del Día')
        ax4.set_ylabel('')
        st.pyplot(fig4)

    # Histograma de Frecuencia Acumulada
    st.divider()
    st.subheader("📊 Distribución Histórica del Índice de Carga Ponderado")
    col_hist, col_text = st.columns([1, 1])

    with col_hist:
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        sns.histplot(df['load_index'], bins=30, cumulative=True, stat="count", color='#16a085', edgecolor='white', alpha=0.8, ax=ax_hist)
        ax_hist.axvline(threshold, color='#e74c3c', linestyle='--', linewidth=2, label=f'Umbral Extremo ({threshold:.1f})')
        ax_hist.set_xlabel('Índice de Carga (Pts)')
        ax_hist.set_ylabel('Frecuencia Acumulada (Horas Totales)')
        ax_hist.legend(loc='lower right')
        st.pyplot(fig_hist)

    with col_text:
        st.info(f"""
        ℹ️ **Efecto de la Ponderación en la Distribución:**
        Al modificar los pesos, este histograma recalcula el índice para toda la historia ({len(df):,} horas). 
        El nuevo umbral dinámico es **{threshold:.1f}**. *(Nota: El Random Forest predice sobre los datos originales)*.
        """)

    # Alertas
    st.divider()
    st.subheader("🚨 Estado Operativo y Protocolos de Contingencia")
    is_alert = any(p == 1 for p in rf_predictions[:3])
    if is_alert:
        st.error(f"""
        **ALERTA PREVENTIVA ACTIVADA (Modelo Random Forest)**
        * El modelo de Machine Learning proyecta **SATURACIÓN INMINENTE** basándose en la presión asistencial actual.
        * Ocupación actual de Boxes: {now_row['boxes_occ']} / {now_row['cap_boxes']} | Censo Hospitalario: {now_row['hospital_census']*100:.1f}%
        
        **📋 PROTOCOLOS SUGERIDOS PARA EJECUCIÓN INMEDIATA:**
        1. Acelerar altas en pisos de hospitalización (Boarding count actual: {int(now_row['boarding_count'])}).
        2. Llamar a personal médico de respaldo.
        3. Habilitar áreas de transición rápida.
        """)
    else:
        st.success(f"""
        **SITUACIÓN ESTABLE**
        * El modelo predictivo no detecta patrones de saturación extrema en el corto plazo.
        * Mantener el flujo operativo estándar.
        """)

# ========================================
# PESTAÑA 2: IMPORTANCIA DE VARIABLES
# ========================================
with tab2:
    st.header("🧠 Importancia de Variables en Random Forest")
    st.markdown("Descubre qué variables tienen mayor peso estadístico para el modelo predictivo al momento de advertir un caso extremo de saturación[cite: 1].")

    # Selector de horizonte para analizar
    horizontes = [f"contingency_extreme_t+{i}" for i in range(1, 7)]
    horizonte_seleccionado = st.selectbox("Selecciona el horizonte de predicción a analizar:", horizontes)

    # Extraer el modelo y las importancias
    rf_model_selected = models[horizonte_seleccionado]
    importancias = rf_model_selected.feature_importances_

    # Crear DataFrame ordenado
    df_importances = pd.DataFrame({
        'Variable': X_encoded.columns,
        'Importancia Relativa': importancias
    }).sort_values(by='Importancia Relativa', ascending=False).reset_index(drop=True)

    col_imp1, col_imp2 = st.columns([2, 1])

    with col_imp1:
        st.subheader(f"Top 10 Variables más influyentes ({horizonte_seleccionado})")
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        
        # Gráfico de barras horizontales
        sns.barplot(data=df_importances.head(10), x='Importancia Relativa', y='Variable', palette='viridis', ax=ax_imp)
        ax_imp.set_title("Top 10 Variables", fontweight='bold')
        ax_imp.set_xlabel("Peso de la Importancia (0 a 1)")
        ax_imp.set_ylabel("")
        
        st.pyplot(fig_imp)

    with col_imp2:
        st.subheader("Cuadro Completo")
        st.markdown("Todas las variables ordenadas por peso:")
        # Mostrar como tabla con formato
        st.dataframe(df_importances.style.format({'Importancia Relativa': '{:.4f}'}), height=400, use_container_width=True)