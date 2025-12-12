
"""
Advanced interactive Streamlit app that automatically loads all CSV files named *_data.csv
from the same folder and provides:
- Data exploration (tables, column info, missing values)
- Interactive filtering by file / columns / date range (auto-detects datetime columns)
- Time-series plotting, histogram, scatter, correlation heatmap (Plotly)
- Basic cleaning utilities and download cleaned CSV
- PCA-based dimensionality reduction visualization (if numeric columns >=3)
"""

import streamlit as st
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Advanced Multi-CSV Explorer", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_csv_files(data_dir=".", pattern="*_data.csv"):
    p = Path(data_dir)
    files = sorted(list(p.glob(pattern)))
    datasets = {}
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            datasets[f.name] = df
        except Exception as e:
            datasets[f.name] = pd.DataFrame({"_load_error": [str(e)]})
    return datasets

def detect_datetime_columns(df):
    dt_cols = []
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            dt_cols.append(c)
        else:
            # try to parse small sample
            try:
                s = pd.to_datetime(df[c], errors="coerce")
                if s.notna().sum() / max(1, len(s)) > 0.5:
                    dt_cols.append(c)
            except:
                pass
    return dt_cols

def main():
    st.title("🔎 Advanced Multi-CSV Data Explorer")
    st.markdown("Automatically discovers `*_data.csv` files in the app folder and provides interactive exploration tools.")
    st.sidebar.header("Configuration")
    data_dir = st.sidebar.text_input("Data directory", value=".")
    pattern = st.sidebar.text_input("Filename pattern", value="*_data.csv")
    datasets = load_csv_files(data_dir, pattern)
    if not datasets:
        st.warning(f"No files found with pattern `{pattern}` in `{data_dir}`.")
        return

    file_names = list(datasets.keys())
    st.sidebar.subheader("Files found")
    selected_files = st.sidebar.multiselect("Select file(s) to work with", options=file_names, default=file_names[:1])

    combine = st.sidebar.checkbox("Combine selected files into one DataFrame (append)", value=False)

    if len(selected_files) == 0:
        st.info("Select at least one file from the sidebar to begin.")
        return

    if combine:
        dfs = []
        for fn in selected_files:
            df = datasets[fn].copy()
            df["_source_file"] = fn
            dfs.append(df)
        try:
            df_all = pd.concat(dfs, ignore_index=True, sort=False)
        except Exception as e:
            st.error(f"Failed to combine: {e}")
            return
    else:
        df_all = datasets[selected_files[0]].copy()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick actions")
    if st.sidebar.button("Show raw (first 10 rows)"):
        st.dataframe(df_all.head(10))

    # Show dataset info
    st.header("Dataset overview")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.metric("Rows", df_all.shape[0])
    with c2:
        st.metric("Columns", df_all.shape[1])
    with c3:
        missing = df_all.isna().sum().sum()
        st.metric("Missing values (total)", int(missing))

    st.subheader("Column types & sample")
    col_info = pd.DataFrame({
        "column": df_all.columns,
        "dtype": [str(df_all[col].dtype) for col in df_all.columns],
        "n_unique": [df_all[col].nunique(dropna=False) for col in df_all.columns],
        "pct_missing": [df_all[col].isna().mean() for col in df_all.columns],
        "sample": [repr(df_all[col].dropna().astype(str).head(3).tolist()) for col in df_all.columns]
    }).sort_values("pct_missing", ascending=False)
    st.dataframe(col_info, use_container_width=True)

    # Allow user to select columns for analysis
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df_all.columns.tolist()

    st.sidebar.subheader("Exploration controls")
    chosen_numeric = st.sidebar.multiselect("Numeric columns (for plots / stats)", options=numeric_cols, default=numeric_cols[:2])
    chosen_x = st.sidebar.selectbox("X axis (for scatter/time)", options=all_cols, index=0)
    chosen_y = st.sidebar.selectbox("Y axis (for scatter)", options=all_cols, index=1 if len(all_cols)>1 else 0)

    # Detect datetime columns and let user use date range filter
    dt_cols = detect_datetime_columns(df_all)
    date_filter_col = None
    if dt_cols:
        date_filter_col = st.sidebar.selectbox("Detected datetime column (for time filtering)", options=["(none)"] + dt_cols, index=0)
    else:
        st.sidebar.info("No obvious datetime column detected.")

    df_work = df_all.copy()

    # If selected datetime col, coerce and enable date range filter
    if date_filter_col and date_filter_col != "(none)":
        df_work[date_filter_col] = pd.to_datetime(df_work[date_filter_col], errors="coerce")
        min_date = df_work[date_filter_col].min()
        max_date = df_work[date_filter_col].max()
        if pd.notna(min_date) and pd.notna(max_date):
            start, end = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()))
            df_work = df_work[(df_work[date_filter_col] >= pd.to_datetime(start)) & (df_work[date_filter_col] <= pd.to_datetime(end))]
        st.write(f"Filtered rows after date selection: {df_work.shape[0]}")

    # Column filter
    chosen_display_cols = st.sidebar.multiselect("Columns to display (table)", options=all_cols, default=all_cols[:min(10, len(all_cols))])
    if chosen_display_cols:
        st.dataframe(df_work[chosen_display_cols].head(500), use_container_width=True)
    else:
        st.dataframe(df_work.head(200), use_container_width=True)

    # Summary statistics
    st.header("Summary statistics & distributions")
    if chosen_numeric:
        st.subheader("Descriptive statistics (selected numeric columns)")
        st.dataframe(df_work[chosen_numeric].describe().T, use_container_width=True)

        st.subheader("Histograms")
        hist_col = st.selectbox("Choose numeric column for histogram", options=chosen_numeric, index=0)
        bins = st.slider("Bins", min_value=5, max_value=200, value=40)
        fig_hist = px.histogram(df_work, x=hist_col, nbins=bins, title=f"Histogram of {hist_col}")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Time-series plotting (if datetime available)")
        if date_filter_col and date_filter_col != "(none)":
            ts_col = st.selectbox("Numeric column to plot over time", options=chosen_numeric, index=0)
            df_ts = df_work[[date_filter_col, ts_col]].dropna()
            fig_ts = px.line(df_ts.sort_values(date_filter_col), x=date_filter_col, y=ts_col, title=f"{ts_col} over time")
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No datetime column selected — time-series plot hidden.")

        st.subheader("Scatter plot")
        scatter_x = chosen_x
        scatter_y = chosen_y
        fig_scatter = px.scatter(df_work, x=scatter_x, y=scatter_y, title=f"{scatter_y} vs {scatter_x}", hover_data=df_work.columns.tolist())
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No numeric columns detected in the selected dataset(s).")

    # Correlation heatmap
    st.header("Correlation & relationships")
    if len(numeric_cols) >= 2:
        corr = df_work[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation matrix.")

    # PCA visualization
    st.header("Dimensionality reduction (PCA)")
    if len(numeric_cols) >= 3:
        n_comp = st.slider("Number of PCA components to compute (2 or 3)", min_value=2, max_value=3, value=2)
        df_num = df_work[numeric_cols].dropna()
        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(df_num)
        pca = PCA(n_components=n_comp)
        Xp = pca.fit_transform(X)
        df_pca = pd.DataFrame(Xp, columns=[f"PC{i+1}" for i in range(n_comp)])
        if "_source_file" in df_work.columns:
            # align index
            df_pca["_source_file"] = df_work.loc[df_num.index, "_source_file"].values
        if n_comp == 2:
            fig_pca = px.scatter(df_pca, x="PC1", y="PC2", color=df_pca.columns[-1] if "_source_file" in df_pca.columns else None,
                                 title="PCA (2 components)")
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            fig_pca = px.scatter_3d(df_pca, x="PC1", y="PC2", z="PC3", color=df_pca.columns[-1] if "_source_file" in df_pca.columns else None,
                                    title="PCA (3 components)")
            st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("Need at least 3 numeric columns for PCA.")

    # Missing values and cleaning utilities
    st.header("Missing values and basic cleaning")
    missing_by_col = df_all.isna().sum().sort_values(ascending=False)
    st.dataframe(missing_by_col, use_container_width=True)

    st.subheader("Basic cleaning options")
    cleaning_action = st.selectbox("Choose cleaning action", options=[
        "None",
        "Drop rows with > X% missing",
        "Fill numeric NA with mean",
        "Fill categorical NA with mode"
    ])
    df_clean = df_all.copy()
    if cleaning_action == "Drop rows with > X% missing":
        pct = st.slider("Max percent missing allowed", min_value=0, max_value=100, value=50)
        thresh = int((1 - pct/100.0) * df_clean.shape[1])
        df_clean = df_clean.dropna(thresh=thresh)
        st.write(f"Rows after drop: {df_clean.shape[0]}")
    elif cleaning_action == "Fill numeric NA with mean":
        for c in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[c] = df_clean[c].fillna(df_clean[c].mean())
        st.write("Filled numeric NAs with column mean.")
    elif cleaning_action == "Fill categorical NA with mode":
        for c in df_clean.select_dtypes(exclude=[np.number]).columns:
            if not df_clean[c].mode().empty:
                df_clean[c] = df_clean[c].fillna(df_clean[c].mode().iloc[0])
        st.write("Filled categorical NAs with mode.")

    st.download_button("Download cleaned CSV", data=df_clean.to_csv(index=False).encode("utf-8"), file_name="cleaned_data.csv")

    # Dataset preview & quick search
    st.header("Search & filter rows (interactive)")
    query_col = st.selectbox("Column to query (text search)", options=all_cols)
    query_text = st.text_input("Text to search (substring match)")
    if query_text:
        mask = df_work[query_col].astype(str).str.contains(query_text, case=False, na=False)
        st.dataframe(df_work.loc[mask].head(500), use_container_width=True)

    st.markdown("---")
    st.caption("This app is auto-generated. Open-source and adjustable — edit `streamlit_app.py` as needed.")
    st.info("To run this app locally: `streamlit run streamlit_app.py` in the folder containing your CSV files.")

if __name__ == '__main__':
    main()
