import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

st.title("Dashboard de Ventas")
st.write("Aplicación desarrollada para el TP de Ingeniería del Software.")

COLUMNAS_REQUERIDAS = [
    "InvoiceNo",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "Country"
]

# =========================
# FUNCIONES DE CARGA
# =========================
@st.cache_data
def cargar_datos_default():
    url = "https://github.com/zetkhho/IdS_tp_grupo8/raw/refs/heads/main/data/Online%20Retail.xlsx"
    df = pd.read_excel(url)
    return df

def validar_columnas(df):
    faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    return faltantes

def leer_archivo_subido(archivo):
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)
    return df

# =========================
# PREPARACIÓN DE DATOS
# =========================
@st.cache_data
def preparar_datos(df):
    df_prep = df.copy()

    # Asegurar formato datetime
    df_prep["InvoiceDate"] = pd.to_datetime(df_prep["InvoiceDate"], errors="coerce")

    # Eliminar fechas inválidas
    df_prep = df_prep.dropna(subset=["InvoiceDate"])

    # Eliminar duplicados
    df_prep = df_prep.drop_duplicates()

    # Eliminar filas sin descripción
    df_prep = df_prep.dropna(subset=["Description"])

    # Filtrar ventas válidas
    df_prep = df_prep[(df_prep["Quantity"] > 0) & (df_prep["UnitPrice"] > 0)]

    # Variables derivadas
    df_prep["TotalAmount"] = df_prep["Quantity"] * df_prep["UnitPrice"]
    df_prep["Year"] = df_prep["InvoiceDate"].dt.year
    df_prep["Month"] = df_prep["InvoiceDate"].dt.month
    df_prep["Day"] = df_prep["InvoiceDate"].dt.day
    df_prep["YearMonth"] = df_prep["InvoiceDate"].dt.to_period("M").astype(str)

    # Excluir conceptos no comerciales para modelado
    patron_exclusion = "POSTAGE|Manual|AMAZON FEE|Adjust bad debt|DOTCOM"
    df_model = df_prep[
        ~df_prep["Description"].str.contains(patron_exclusion, case=False, na=False)
    ].copy()

    return df_prep, df_model

# =========================
# MODELADO
# =========================
@st.cache_data
def construir_modelo(df_model, grado=3):
    ventas_mensuales = df_model.groupby("YearMonth")["TotalAmount"].sum().reset_index()
    ventas_mensuales["YearMonth"] = pd.to_datetime(ventas_mensuales["YearMonth"])
    ventas_mensuales = ventas_mensuales.sort_values("YearMonth").reset_index(drop=True)
    ventas_mensuales["TimeIndex"] = np.arange(len(ventas_mensuales))

    X = ventas_mensuales[["TimeIndex"]]
    y = ventas_mensuales["TotalAmount"]

    poly = PolynomialFeatures(degree=grado, include_bias=False)
    X_poly = poly.fit_transform(X)

    modelo_poly = LinearRegression()
    modelo_poly.fit(X_poly, y)

    predicciones_poly = modelo_poly.predict(X_poly)
    ventas_mensuales["Predicciones_Polinomicas"] = np.round(predicciones_poly, 2)

    futuros = pd.DataFrame({
        "TimeIndex": np.arange(len(ventas_mensuales), len(ventas_mensuales) + 12)
    })
    X_futuro_poly = poly.transform(futuros)
    pred_futuras = modelo_poly.predict(X_futuro_poly)

    ultimo_mes = ventas_mensuales["YearMonth"].max()
    meses_futuros = pd.date_range(
        start=ultimo_mes + pd.offsets.MonthBegin(1),
        periods=12,
        freq="MS"
    )

    df_futuro = pd.DataFrame({
        "YearMonth": meses_futuros,
        "Predicciones": np.round(pred_futuras, 2)
    })

    # Evitar mostrar valores negativos en la app
    df_futuro["Predicciones"] = np.maximum(df_futuro["Predicciones"], 0)

    return ventas_mensuales, df_futuro

# =========================
# SIDEBAR / CARGA OPCIONAL
# =========================
st.sidebar.header("Carga de datos")
archivo = st.sidebar.file_uploader(
    "Subí un archivo CSV o Excel",
    type=["csv", "xlsx"]
)

if archivo is not None:
    try:
        df = leer_archivo_subido(archivo)
        faltantes = validar_columnas(df)

        if faltantes:
            st.error(f"El archivo no tiene las columnas necesarias: {faltantes}")
            st.stop()

        st.sidebar.success("Archivo cargado correctamente.")
        fuente_datos = f"Archivo subido: {archivo.name}"
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()
else:
    df = cargar_datos_default()
    fuente_datos = "Dataset Online Retail del repositorio"
    st.sidebar.info("Usando dataset Online Retail del repositorio.")

st.sidebar.write("**Fuente actual:**")
st.sidebar.write(fuente_datos)

# =========================
# EJECUCIÓN
# =========================
df_prep, df_model = preparar_datos(df)
ventas_mensuales, df_futuro = construir_modelo(df_model, grado=3)

# =========================
# KPIs
# =========================
st.subheader("Indicadores generales")

col1, col2, col3, col4 = st.columns(4)

total_vendido = df_prep["TotalAmount"].sum()
transacciones = df_prep["InvoiceNo"].nunique()
productos = df_model["Description"].nunique()
paises = df_prep["Country"].nunique()

col1.metric("Monto total vendido", f"{total_vendido:,.2f}")
col2.metric("Cantidad de transacciones", f"{transacciones:,}")
col3.metric("Productos distintos", f"{productos:,}")
col4.metric("Países", f"{paises:,}")

# =========================
# EVOLUCIÓN MENSUAL
# =========================
st.subheader("Evolución mensual de ventas")

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(
    ventas_mensuales["YearMonth"],
    ventas_mensuales["TotalAmount"],
    marker="o",
    label="Valores reales"
)
ax.plot(
    ventas_mensuales["YearMonth"],
    ventas_mensuales["Predicciones_Polinomicas"],
    marker="o",
    label="Modelo polinómico grado 3"
)
ax.plot(
    df_futuro["YearMonth"],
    df_futuro["Predicciones"],
    marker="o",
    linestyle="--",
    label="Predicción futura"
)
ax.set_title("Serie temporal de ventas mensuales")
ax.set_xlabel("Fecha")
ax.set_ylabel("Monto total")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# =========================
# TOP PRODUCTOS POR MONTO
# =========================
st.subheader("Top 10 productos por monto vendido")

top_productos = (
    df_model.groupby("Description")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

fig2, ax2 = plt.subplots(figsize=(8, 4))
top_productos.plot(kind="barh", ax=ax2)
ax2.set_title("Top 10 productos por monto vendido")
ax2.set_xlabel("Monto total vendido")
ax2.set_ylabel("Producto")
plt.tight_layout()
st.pyplot(fig2)

# =========================
# TOP PRODUCTOS POR CANTIDAD
# =========================
st.subheader("Top 10 productos por cantidad vendida")

top_productos_qty = (
    df_model.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

fig3, ax3 = plt.subplots(figsize=(8, 4))
top_productos_qty.plot(kind="barh", ax=ax3)
ax3.set_title("Top 10 productos por cantidad vendida")
ax3.set_xlabel("Cantidad vendida")
ax3.set_ylabel("Producto")
plt.tight_layout()
st.pyplot(fig3)

# =========================
# TOP PAÍSES
# =========================
st.subheader("Top 10 países por monto vendido")

top_paises = (
    df_prep.groupby("Country")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

fig4, ax4 = plt.subplots(figsize=(8, 4))
top_paises.plot(kind="barh", ax=ax4)
ax4.set_title("Top 10 países por monto vendido")
ax4.set_xlabel("Monto total vendido")
ax4.set_ylabel("País")
plt.tight_layout()
st.pyplot(fig4)

# =========================
# TABLA DE PREDICCIONES
# =========================
st.subheader("Predicción para los próximos 12 meses")

df_futuro_mostrar = df_futuro.copy()
df_futuro_mostrar["YearMonth"] = df_futuro_mostrar["YearMonth"].dt.strftime("%Y-%m")
df_futuro_mostrar["Predicciones"] = df_futuro_mostrar["Predicciones"].round(2)

st.dataframe(df_futuro_mostrar, use_container_width=True)

# =========================
# DETALLE DEL PROYECTO
# =========================
with st.expander("Ver detalle de preparación de datos"):
    st.write("Dimensión original:", df.shape)
    st.write("Dimensión luego de preparación:", df_prep.shape)
    st.write("Dimensión usada para modelado:", df_model.shape)
    st.write("Fuente de datos:", fuente_datos)
