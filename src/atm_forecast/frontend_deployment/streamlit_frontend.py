"""
AtmosNet — Streamlit Frontend
==============================

Full-featured Streamlit dashboard for the AtmosNet multi-target
atmospheric forecasting system.

Pages
-----
1. **Forecast** — single / multi-target forecast with model selection,
   location filter, configurable horizon.  Line chart + data table.
2. **Compare Targets** — overlay / subplot comparison of 2-7 targets.
3. **Relationship Analysis** — scatter plots, correlation heatmaps,
   distribution charts (histogram / box / violin).
4. **Geographic View** — per-country bar charts, scatter-mapbox
   visualisation of forecasts on a world map.
5. **Model Explorer** — side-by-side model metrics, per-target R²
   breakdown, training curves.

Data flow
---------
Raw lake data → PreprocessingPipeline (clean) → feature engineering →
transform → create_sequences → model.predict → inverse_transform →
visualise in Streamlit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

logger = logging.getLogger(__name__)

# =====================================================================
# Constants
# =====================================================================

TARGETS: List[str] = [
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
    "temperature_celsius",
]

TARGET_DISPLAY: Dict[str, str] = {
    "air_quality_Carbon_Monoxide": "Carbon Monoxide (CO)",
    "air_quality_Ozone": "Ozone (O\u2083)",
    "air_quality_Nitrogen_dioxide": "Nitrogen Dioxide (NO\u2082)",
    "air_quality_Sulphur_dioxide": "Sulphur Dioxide (SO\u2082)",
    "air_quality_PM2.5": "PM 2.5",
    "air_quality_PM10": "PM 10",
    "temperature_celsius": "Temperature (\u00b0C)",
}

TARGET_UNITS: Dict[str, str] = {
    "air_quality_Carbon_Monoxide": "\u00b5g/m\u00b3",
    "air_quality_Ozone": "\u00b5g/m\u00b3",
    "air_quality_Nitrogen_dioxide": "\u00b5g/m\u00b3",
    "air_quality_Sulphur_dioxide": "\u00b5g/m\u00b3",
    "air_quality_PM2.5": "\u00b5g/m\u00b3",
    "air_quality_PM10": "\u00b5g/m\u00b3",
    "temperature_celsius": "\u00b0C",
}

MODEL_NAMES = {"bilstm": "Bi-LSTM", "tcn": "TCN", "tft": "TFT"}


# =====================================================================
# Helpers — Data & Model Loading (cached)
# =====================================================================

def _project_root() -> Path:
    """Resolve the project root directory."""
    return Path(__file__).resolve().parents[3]


@st.cache_resource(show_spinner="Loading preprocessing pipeline...")
def load_pipeline(artifacts_dir: Path):
    """Load the fitted PreprocessingPipeline from disk."""
    from atm_forecast.data.preprocessing import PreprocessingPipeline
    return PreprocessingPipeline.load(artifacts_dir)


@st.cache_resource(show_spinner="Loading model...")
def load_keras_model(model_dir: Path):
    """Load a trained Keras model + metadata from disk."""
    from atm_forecast.models.registry import load_model
    model, _scaler, metadata = load_model(model_dir, load_scaler=False)
    return model, metadata


@st.cache_data(show_spinner="Loading raw data...", ttl=3600)
def load_raw_data(lake_raw: str) -> pd.DataFrame:
    """Load raw Parquet from the data lake and return with location metadata intact."""
    df = pd.read_parquet(lake_raw)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"].astype(str))
    return df


@st.cache_data(show_spinner="Preparing data for inference...", ttl=3600)
def prepare_inference_data(
    lake_raw: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data, clean, and run feature engineering.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned data (before feature engineering) — keeps country/location.
    df_engineered : pd.DataFrame
        Fully engineered data (ready for scaling / sequencing).
    """
    from atm_forecast.data.preprocessing import PreprocessingPipeline
    from atm_forecast.features.feature_engineering import run_feature_engineering

    pipe = PreprocessingPipeline(targets=TARGETS)
    df = pipe.load_raw(lake_raw)
    df_clean = pipe.clean(df)
    df_engineered = run_feature_engineering(df_clean.copy(), targets=TARGETS)
    return df_clean, df_engineered


def _get_available_models(artifacts_dir: Path) -> Dict[str, Path]:
    """Scan artifacts/models/ for available trained models."""
    models_dir = artifacts_dir / "models"
    found = {}
    if models_dir.exists():
        for child in models_dir.iterdir():
            if child.is_dir() and (child / "model.keras").exists():
                found[child.name] = child
    return found


def _load_model_metadata(model_dir: Path) -> dict:
    """Load metadata.json for a trained model."""
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _filter_engineered_by_location(
    df_clean: pd.DataFrame,
    df_engineered: pd.DataFrame,
    country: str = "All",
    location: str = "All",
) -> pd.DataFrame:
    """Filter the engineered DataFrame by country / location.

    Uses the ``location_id`` column that was assigned alphabetically
    from ``location_name`` during feature engineering.
    """
    if country == "All" and location == "All":
        return df_engineered

    if "location_id" not in df_engineered.columns:
        return df_engineered  # no location encoding available

    # Reconstruct location_name → location_id mapping (alphabetical)
    if "location_name" not in df_clean.columns:
        return df_engineered

    location_names_sorted = sorted(df_clean["location_name"].unique())
    name_to_id = {name: idx for idx, name in enumerate(location_names_sorted)}

    if location != "All":
        loc_id = name_to_id.get(location)
        if loc_id is not None:
            return df_engineered[df_engineered["location_id"] == loc_id].copy()
        return df_engineered  # location not found, return all

    if country != "All" and "country" in df_clean.columns:
        country_locs = df_clean.loc[
            df_clean["country"] == country, "location_name"
        ].unique()
        loc_ids = [name_to_id[loc] for loc in country_locs if loc in name_to_id]
        if loc_ids:
            return df_engineered[df_engineered["location_id"].isin(loc_ids)].copy()

    return df_engineered


def _run_forecast(
    model,
    pipeline,
    df_engineered: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    n_days: int,
) -> pd.DataFrame:
    """Run model inference and return forecasted values in original scale.

    Parameters
    ----------
    model : keras.Model
    pipeline : PreprocessingPipeline
    df_engineered : pd.DataFrame
        Fully engineered DataFrame (targets + features, unscaled).
    feature_cols : list[str]
        Feature columns from the pipeline.
    seq_len : int
        Model sequence length.
    n_days : int
        Number of forecast steps to return.

    Returns
    -------
    pd.DataFrame
        Columns = TARGETS, rows = forecast steps.
    """
    from atm_forecast.data.preprocessing import create_sequences

    # Scale using the fitted pipeline
    df_scaled = pipeline.transform(df_engineered)

    # Create sequences from the tail of the data
    X, _ = create_sequences(
        df_scaled, seq_len, forecast_h=1,
        feature_cols=feature_cols, target_cols=TARGETS,
    )

    if len(X) == 0:
        return pd.DataFrame(columns=TARGETS)

    # Take the last n_days sequences (most recent)
    X_last = X[-n_days:]
    preds_scaled = model.predict(X_last, verbose=0)
    preds_original = pipeline.inverse_transform_targets(preds_scaled)

    return pd.DataFrame(preds_original, columns=TARGETS)


# =====================================================================
# StreamlitFrontend
# =====================================================================

class StreamlitFrontend:
    """Main Streamlit application for AtmosNet forecasting dashboard.

    Encapsulates all pages and common state.  Launch via::

        app = StreamlitFrontend()
        app.run()
    """

    PAGES = {
        "Forecast": "page_forecast",
        "Compare Targets": "page_compare",
        "Relationship Analysis": "page_relationship",
        "Geographic View": "page_geographic",
        "Model Explorer": "page_model_explorer",
    }

    def __init__(self):
        self.root = _project_root()
        self.artifacts_dir = self.root / "artifacts"
        self.lake_raw = self.root / "data" / "lake" / "raw"
        self.preprocess_dir = self.artifacts_dir / "preprocessing"

    # ─────────────────────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────────────────────

    def run(self):
        """Entry-point: configure page and render the selected page."""
        st.set_page_config(
            page_title="AtmosNet Forecast Dashboard",
            page_icon="\U0001F321\uFE0F",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        self._apply_custom_css()
        self._render_sidebar()

        page_method = self.PAGES.get(st.session_state.get("page", "Forecast"), "page_forecast")
        getattr(self, page_method)()

    # ─────────────────────────────────────────────────────────────
    # Sidebar
    # ─────────────────────────────────────────────────────────────

    def _render_sidebar(self):
        with st.sidebar:
            st.image(
                "https://img.icons8.com/fluency/96/partly-cloudy-day.png",
                width=64,
            )
            st.title("AtmosNet")
            st.caption("Multi-target atmospheric forecasting")
            st.divider()

            page = st.radio(
                "Navigate",
                list(self.PAGES.keys()),
                index=0,
                key="page",
            )

            st.divider()

            # ── Model selector (global) ──────────────────────────
            available = _get_available_models(self.artifacts_dir)
            if available:
                model_key = st.selectbox(
                    "Active model",
                    list(available.keys()),
                    format_func=lambda k: MODEL_NAMES.get(k, k.upper()),
                    key="selected_model",
                )
                st.session_state["model_dir"] = str(available[model_key])
            else:
                st.warning("No trained models found in artifacts/models/")
                st.session_state["model_dir"] = None

            st.divider()
            st.caption("Built with Streamlit + Plotly")

    # ─────────────────────────────────────────────────────────────
    # CSS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_custom_css():
        st.markdown(
            """
            <style>
            /* Metric cards with subtle gradient border */
            [data-testid="stMetric"] {
                background: linear-gradient(135deg, rgba(28, 131, 225, 0.06), rgba(28, 225, 131, 0.04));
                border-radius: 10px;
                padding: 14px 18px;
                border-left: 3px solid rgba(28, 131, 225, 0.4);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            [data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }
            /* Tighter chart spacing */
            .stPlotlyChart { margin-top: -0.5rem; }
            /* Tab labels */
            .stTabs [data-baseweb="tab-list"] button {
                font-weight: 600;
                font-size: 0.95rem;
            }
            /* Expander headers */
            .streamlit-expanderHeader { font-weight: 600; }
            /* Color-coded badge for location chips */
            .loc-badge {
                display: inline-block;
                padding: 2px 10px;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 500;
                margin: 2px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # =================================================================
    # PAGE 1 — Forecast
    # =================================================================

    def page_forecast(self):
        st.header("\U0001F4C8 Forecast")
        st.markdown("Select targets, locations, and time horizon to generate and compare forecasts.")

        model_dir = st.session_state.get("model_dir")
        if model_dir is None:
            st.error("No model selected. Train a model first.")
            return

        # ── Controls ─────────────────────────────────────────────
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
        with col_ctrl1:
            selected_targets = st.multiselect(
                "Targets to forecast",
                TARGETS,
                default=["temperature_celsius"],
                format_func=lambda t: TARGET_DISPLAY.get(t, t),
                key="fc_targets",
            )
        with col_ctrl2:
            n_days = st.slider("Forecast horizon (days)", 1, 30, 7, key="fc_horizon")
        with col_ctrl3:
            chart_type = st.radio("Display", ["Line Chart", "Data Table", "Both"], key="fc_display")

        if not selected_targets:
            st.info("Please select at least one target.")
            return

        # ── Location filter — now supports multi-select ──────────
        try:
            raw_df = load_raw_data(str(self.lake_raw))
            locations = sorted(raw_df["location_name"].unique()) if "location_name" in raw_df.columns else []
            countries = sorted(raw_df["country"].unique()) if "country" in raw_df.columns else []
        except Exception:
            locations, countries = [], []
            raw_df = None

        with st.expander("Location filter", expanded=True):
            sel_country = st.selectbox("Country", ["All"] + countries, key="fc_country")
            if sel_country != "All" and raw_df is not None:
                loc_options = sorted(
                    raw_df.loc[raw_df["country"] == sel_country, "location_name"].unique()
                )
            else:
                loc_options = locations
            sel_locations = st.multiselect(
                "Locations (select multiple to compare)",
                loc_options,
                default=[],
                key="fc_locations",
                help="Leave empty to use all data; select multiple to compare trends across locations.",
            )

        # ── Run inference ────────────────────────────────────────
        pipeline = None
        model = None
        metadata = None
        try:
            pipeline = load_pipeline(self.preprocess_dir)
            model, metadata = load_keras_model(Path(model_dir))
        except FileNotFoundError as e:
            st.error(f"Missing artefacts: {e}. Run the training pipeline first.")
            return
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

        seq_len = metadata.get("seq_len", 24)
        df_clean, df_eng = prepare_inference_data(str(self.lake_raw))

        # Single location or all data  → simple path
        # Multiple locations            → per-location forecasts for comparison
        forecast_locations = sel_locations if sel_locations else ["All"]
        multi_location = len(forecast_locations) > 1

        location_forecasts: Dict[str, pd.DataFrame] = {}
        with st.spinner("Running forecasts..."):
            progress = st.progress(0) if multi_location else None
            for i, loc in enumerate(forecast_locations):
                try:
                    df_filtered = _filter_engineered_by_location(
                        df_clean, df_eng,
                        country=sel_country if loc == "All" else "All",
                        location=loc if loc != "All" else "All",
                    )
                    fc = _run_forecast(
                        model, pipeline, df_filtered,
                        pipeline.feature_cols, seq_len, n_days,
                    )
                    if not fc.empty:
                        fc.index = pd.RangeIndex(1, len(fc) + 1, name="Day")
                        location_forecasts[loc] = fc
                except Exception as exc:
                    logger.warning("Forecast failed for %s: %s", loc, exc)
                if progress:
                    progress.progress((i + 1) / len(forecast_locations))
            if progress:
                progress.empty()

        if not location_forecasts:
            st.warning("Not enough data to generate forecasts.")
            return

        # ── Metric cards (use first / only location) ─────────────
        first_loc = list(location_forecasts.keys())[0]
        first_fc = location_forecasts[first_loc]
        metric_cols = st.columns(min(len(selected_targets), 4))
        for i, tgt in enumerate(selected_targets):
            col = metric_cols[i % len(metric_cols)]
            vals = first_fc[tgt]
            label = TARGET_DISPLAY.get(tgt, tgt)
            unit = TARGET_UNITS.get(tgt, "")
            loc_label = f" ({first_loc})" if multi_location else ""
            col.metric(
                label=f"{label}{loc_label}",
                value=f"{vals.mean():.2f} {unit}",
                delta=f"min {vals.min():.2f} / max {vals.max():.2f}",
            )

        # ── Model info badge ─────────────────────────────────────
        model_name = metadata.get("model_name", "unknown")
        avg_r2 = metadata.get("avg_r2", None)
        badge = f"**Model:** {MODEL_NAMES.get(model_name, model_name)}"
        if avg_r2 is not None:
            badge += f" | **Test R\u00b2:** {avg_r2:.4f}"
        if multi_location:
            badge += f" | **Locations:** {len(location_forecasts)}"
        st.caption(badge)

        # ── Visualisation ────────────────────────────────────────
        if not multi_location:
            # Single location — simple view
            display_df = first_fc[selected_targets].rename(columns=TARGET_DISPLAY)

            if chart_type in ("Line Chart", "Both"):
                fig = px.line(
                    display_df,
                    x=display_df.index,
                    y=display_df.columns,
                    markers=True,
                    title="Forecasted Values",
                    labels={"value": "Value", "variable": "Target", "x": "Day"},
                )
                fig.update_layout(
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

            if chart_type in ("Data Table", "Both"):
                st.dataframe(display_df.style.format("{:.4f}"), use_container_width=True)
        else:
            # Multi-location — one chart per target with location-coloured lines
            tab_chart, tab_table, tab_stats = st.tabs(
                ["\U0001F4C8 Trend Charts", "\U0001F4CB Data Table", "\U0001F4CA Statistics"]
            )

            with tab_chart:
                for tgt in selected_targets:
                    tgt_label = TARGET_DISPLAY.get(tgt, tgt)
                    tgt_unit = TARGET_UNITS.get(tgt, "")
                    fig = go.Figure()
                    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
                    for j, (loc, fc_df) in enumerate(location_forecasts.items()):
                        fig.add_trace(go.Scatter(
                            x=list(fc_df.index),
                            y=fc_df[tgt],
                            mode="lines+markers",
                            name=loc,
                            line=dict(color=colors[j % len(colors)], width=2),
                            marker=dict(size=6),
                            hovertemplate=f"<b>{loc}</b><br>Day %{{x}}<br>{tgt_label}: %{{y:.3f}} {tgt_unit}<extra></extra>",
                        ))
                    fig.update_layout(
                        title=f"{tgt_label} — by Location",
                        xaxis_title="Day",
                        yaxis_title=f"{tgt_label} ({tgt_unit})",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        template="plotly_white",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab_table:
                # Build combined table: Day × (Location, Target)
                rows = []
                for loc, fc_df in location_forecasts.items():
                    for day in fc_df.index:
                        row = {"Day": day, "Location": loc}
                        for tgt in selected_targets:
                            row[TARGET_DISPLAY.get(tgt, tgt)] = fc_df.loc[day, tgt]
                        rows.append(row)
                combined_df = pd.DataFrame(rows)
                st.dataframe(
                    combined_df.style.format(
                        {c: "{:.4f}" for c in combined_df.columns if c not in ("Day", "Location")}
                    ),
                    use_container_width=True,
                )

            with tab_stats:
                # Summary statistics per location per target
                stat_rows = []
                for loc, fc_df in location_forecasts.items():
                    row = {"Location": loc}
                    for tgt in selected_targets:
                        label = TARGET_DISPLAY.get(tgt, tgt)
                        vals = fc_df[tgt]
                        row[f"{label} Mean"] = vals.mean()
                        row[f"{label} Min"] = vals.min()
                        row[f"{label} Max"] = vals.max()
                        row[f"{label} Std"] = vals.std()
                    stat_rows.append(row)
                stats_df = pd.DataFrame(stat_rows)
                st.dataframe(
                    stats_df.style.format(
                        {c: "{:.4f}" for c in stats_df.columns if c != "Location"}
                    ).background_gradient(axis=0, subset=[c for c in stats_df.columns if "Mean" in c]),
                    use_container_width=True,
                )

        # ── Download ─────────────────────────────────────────────
        if multi_location:
            csv = combined_df.to_csv(index=False)
        else:
            csv = display_df.to_csv()
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="forecast.csv",
            mime="text/csv",
        )

    # =================================================================
    # PAGE 2 — Compare Targets
    # =================================================================

    def page_compare(self):
        st.header("\U0001F504 Compare Targets")
        st.markdown("Compare forecasted values across multiple targets side-by-side.")

        model_dir = st.session_state.get("model_dir")
        if model_dir is None:
            st.error("No model selected.")
            return

        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.multiselect(
                "Select 2-7 targets to compare",
                TARGETS,
                default=["temperature_celsius", "air_quality_Ozone"],
                format_func=lambda t: TARGET_DISPLAY.get(t, t),
                key="cmp_targets",
                max_selections=7,
            )
        with col2:
            n_days = st.slider("Horizon (days)", 1, 30, 7, key="cmp_horizon")

        if len(selected) < 2:
            st.info("Select at least 2 targets to compare.")
            return

        view_mode = st.radio(
            "Layout", ["Overlay", "Subplots", "Difference"],
            horizontal=True, key="cmp_layout",
        )

        # ── Generate forecasts ───────────────────────────────────
        with st.spinner("Running forecast..."):
            try:
                pipeline = load_pipeline(self.preprocess_dir)
                model, metadata = load_keras_model(Path(model_dir))
                seq_len = metadata.get("seq_len", 24)
                _, df_eng = prepare_inference_data(str(self.lake_raw))
                forecast_df = _run_forecast(
                    model, pipeline, df_eng,
                    pipeline.feature_cols, seq_len, n_days,
                )
                if forecast_df.empty:
                    st.warning("Not enough data.")
                    return
                forecast_df.index = pd.RangeIndex(1, len(forecast_df) + 1, name="Day")
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                return

        display_df = forecast_df[selected].rename(columns=TARGET_DISPLAY)

        if view_mode == "Overlay":
            fig = px.line(
                display_df, x=display_df.index, y=display_df.columns,
                markers=True, title="Target Comparison — Overlay",
                labels={"value": "Value", "variable": "Target", "x": "Day"},
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        elif view_mode == "Subplots":
            fig = make_subplots(
                rows=len(selected), cols=1,
                shared_xaxes=True,
                subplot_titles=[TARGET_DISPLAY.get(t, t) for t in selected],
                vertical_spacing=0.04,
            )
            colors = px.colors.qualitative.Set2
            for i, tgt in enumerate(selected):
                disp = TARGET_DISPLAY.get(tgt, tgt)
                fig.add_trace(
                    go.Scatter(
                        x=list(display_df.index),
                        y=forecast_df[tgt],
                        mode="lines+markers",
                        name=disp,
                        line=dict(color=colors[i % len(colors)]),
                    ),
                    row=i + 1, col=1,
                )
            fig.update_layout(
                height=250 * len(selected),
                title_text="Target Comparison — Subplots",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif view_mode == "Difference":
            st.markdown("**Difference chart:** first selected target minus second.")
            if len(selected) >= 2:
                diff = forecast_df[selected[0]] - forecast_df[selected[1]]
                diff_df = pd.DataFrame({
                    "Day": range(1, len(diff) + 1),
                    f"{TARGET_DISPLAY[selected[0]]} - {TARGET_DISPLAY[selected[1]]}": diff.values,
                })
                fig = px.bar(
                    diff_df, x="Day",
                    y=diff_df.columns[1],
                    title="Difference (Target A \u2212 Target B)",
                    color=diff_df.columns[1],
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Summary stats table ──────────────────────────────────
        with st.expander("Summary statistics"):
            stats_df = display_df.describe().T
            st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)

        csv = display_df.to_csv()
        st.download_button("Download CSV", data=csv, file_name="comparison.csv", mime="text/csv")

    # =================================================================
    # PAGE 3 — Relationship Analysis
    # =================================================================

    def page_relationship(self):
        st.header("\U0001F50D Relationship Analysis")
        st.markdown("Explore correlations, scatter plots, and distributions across targets.")

        model_dir = st.session_state.get("model_dir")
        if model_dir is None:
            st.error("No model selected.")
            return

        n_days = st.slider("Forecast horizon (days)", 7, 30, 14, key="rel_horizon")

        with st.spinner("Running forecast..."):
            try:
                pipeline = load_pipeline(self.preprocess_dir)
                model, metadata = load_keras_model(Path(model_dir))
                seq_len = metadata.get("seq_len", 24)
                _, df_eng = prepare_inference_data(str(self.lake_raw))
                forecast_df = _run_forecast(
                    model, pipeline, df_eng,
                    pipeline.feature_cols, seq_len, n_days,
                )
                if forecast_df.empty:
                    st.warning("Not enough data.")
                    return
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                return

        tab_scatter, tab_corr, tab_dist, tab_pair = st.tabs(
            ["Scatter Plot", "Correlation Heatmap", "Distribution", "Pair Plot"]
        )

        # ── Scatter Plot ─────────────────────────────────────────
        with tab_scatter:
            sc1, sc2 = st.columns(2)
            with sc1:
                x_target = st.selectbox(
                    "X-axis target", TARGETS,
                    format_func=lambda t: TARGET_DISPLAY.get(t, t),
                    key="sc_x",
                )
            with sc2:
                y_target = st.selectbox(
                    "Y-axis target", TARGETS,
                    index=1 if len(TARGETS) > 1 else 0,
                    format_func=lambda t: TARGET_DISPLAY.get(t, t),
                    key="sc_y",
                )

            fig = px.scatter(
                forecast_df, x=x_target, y=y_target,
                trendline="ols",
                labels={
                    x_target: TARGET_DISPLAY.get(x_target, x_target),
                    y_target: TARGET_DISPLAY.get(y_target, y_target),
                },
                title=f"{TARGET_DISPLAY[x_target]} vs {TARGET_DISPLAY[y_target]}",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Correlation Heatmap ──────────────────────────────────
        with tab_corr:
            corr_method = st.radio(
                "Method", ["pearson", "spearman"], horizontal=True, key="corr_method"
            )
            corr_matrix = forecast_df[TARGETS].corr(method=corr_method)
            display_labels = [TARGET_DISPLAY.get(t, t) for t in TARGETS]

            fig = px.imshow(
                corr_matrix.values,
                x=display_labels, y=display_labels,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=".2f",
                title=f"{corr_method.title()} Correlation Matrix",
                aspect="equal",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        # ── Distribution ─────────────────────────────────────────
        with tab_dist:
            dist_target = st.selectbox(
                "Target", TARGETS,
                format_func=lambda t: TARGET_DISPLAY.get(t, t),
                key="dist_target",
            )
            dist_type = st.radio(
                "Chart type", ["Histogram", "Box Plot", "Violin"],
                horizontal=True, key="dist_type",
            )
            label = TARGET_DISPLAY.get(dist_target, dist_target)

            if dist_type == "Histogram":
                fig = px.histogram(
                    forecast_df, x=dist_target, nbins=20,
                    title=f"Distribution of {label}",
                    labels={dist_target: label},
                    marginal="rug",
                )
            elif dist_type == "Box Plot":
                fig = px.box(
                    forecast_df, y=dist_target,
                    title=f"Box Plot — {label}",
                    labels={dist_target: label},
                    points="all",
                )
            else:
                fig = px.violin(
                    forecast_df, y=dist_target,
                    title=f"Violin Plot — {label}",
                    labels={dist_target: label},
                    box=True, points="all",
                )
            st.plotly_chart(fig, use_container_width=True)

        # ── Pair Plot ────────────────────────────────────────────
        with tab_pair:
            pair_targets = st.multiselect(
                "Select 2-4 targets for pair plot",
                TARGETS,
                default=TARGETS[:3],
                format_func=lambda t: TARGET_DISPLAY.get(t, t),
                key="pair_targets",
                max_selections=4,
            )
            if len(pair_targets) >= 2:
                fig = px.scatter_matrix(
                    forecast_df[pair_targets],
                    dimensions=pair_targets,
                    labels={t: TARGET_DISPLAY.get(t, t) for t in pair_targets},
                    title="Pair Plot Matrix",
                )
                fig.update_layout(height=700)
                fig.update_traces(diagonal_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least 2 targets.")

    # =================================================================
    # PAGE 4 — Geographic View
    # =================================================================

    def page_geographic(self):
        st.header("\U0001F30D Geographic View")
        st.markdown("Visualise forecasts across countries and locations on a map.")

        model_dir = st.session_state.get("model_dir")
        if model_dir is None:
            st.error("No model selected.")
            return

        # ── Load raw data for location metadata ──────────────────
        try:
            raw_df = load_raw_data(str(self.lake_raw))
        except Exception as e:
            st.error(f"Could not load raw data: {e}")
            return

        if "country" not in raw_df.columns or "location_name" not in raw_df.columns:
            st.error("Raw data is missing country / location_name columns.")
            return

        # Build location lookup: location_name → (country, lat, lon)
        loc_meta = (
            raw_df.groupby("location_name")
            .agg(
                country=("country", "first"),
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
            )
            .reset_index()
        )

        # ── Controls ─────────────────────────────────────────────
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            geo_targets = st.multiselect(
                "Targets",
                TARGETS,
                default=["temperature_celsius"],
                format_func=lambda t: TARGET_DISPLAY.get(t, t),
                key="geo_targets",
            )
        with c2:
            n_days = st.slider("Horizon (days)", 1, 30, 7, key="geo_horizon")
        with c3:
            sel_countries = st.multiselect(
                "Countries",
                sorted(loc_meta["country"].unique()),
                default=sorted(loc_meta["country"].unique())[:5],
                key="geo_countries",
            )

        if not geo_targets:
            st.info("Select at least one target.")
            return

        # Filter locations by selected countries
        if sel_countries:
            loc_filtered = loc_meta[loc_meta["country"].isin(sel_countries)]
        else:
            loc_filtered = loc_meta

        # ── Run per-location forecasts ────────────────────────────
        with st.spinner("Running per-location forecasts..."):
            try:
                pipeline = load_pipeline(self.preprocess_dir)
                model, metadata = load_keras_model(Path(model_dir))
                seq_len = metadata.get("seq_len", 24)
                df_clean, df_eng = prepare_inference_data(str(self.lake_raw))
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                return

            # Run forecast per location so each gets unique values
            geo_results = []
            progress = st.progress(0, text="Forecasting...")
            total = len(loc_filtered)
            for idx, (_, row) in enumerate(loc_filtered.iterrows()):
                loc_name = row["location_name"]
                try:
                    df_loc = _filter_engineered_by_location(
                        df_clean, df_eng,
                        country="All", location=loc_name,
                    )
                    if df_loc.empty:
                        continue
                    fc = _run_forecast(
                        model, pipeline, df_loc,
                        pipeline.feature_cols, seq_len, n_days,
                    )
                    if fc.empty:
                        continue
                    loc_means = fc[geo_targets].mean()
                    result = {
                        "location_name": loc_name,
                        "country": row["country"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                    }
                    for tgt in geo_targets:
                        result[TARGET_DISPLAY.get(tgt, tgt)] = loc_means[tgt]
                    geo_results.append(result)
                except Exception:
                    logger.debug("Forecast skipped for %s", loc_name, exc_info=True)
                progress.progress((idx + 1) / total, text=f"{loc_name}")
            progress.empty()

            if not geo_results:
                st.warning("No forecasts could be generated for the selected locations.")
                return

        map_data = pd.DataFrame(geo_results)
        agg_cols = [TARGET_DISPLAY.get(t, t) for t in geo_targets]

        tab_map, tab_bar, tab_radar, tab_table = st.tabs(
            ["\U0001F5FA Map View", "\U0001F4CA Country Bar Chart", "\U0001F3AF Radar", "\U0001F4CB Data Table"]
        )

        # ── Map View — one sub-map per target ────────────────────
        with tab_map:
            if len(geo_targets) == 1:
                # Single target  → one large map
                tgt = geo_targets[0]
                label = TARGET_DISPLAY.get(tgt, tgt)
                fig = px.scatter_mapbox(
                    map_data, lat="latitude", lon="longitude",
                    hover_name="location_name",
                    hover_data={
                        "country": True, label: ":.3f",
                        "latitude": ":.2f", "longitude": ":.2f",
                    },
                    color=label,
                    size=np.abs(map_data[label]).clip(lower=0.1).tolist(),
                    color_continuous_scale="Turbo",
                    mapbox_style="carto-positron",
                    zoom=1, height=600,
                    title=f"Forecast Map — {label}",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Multiple targets → one map per target in columns
                cols_per_row = min(len(geo_targets), 2)
                for row_start in range(0, len(geo_targets), cols_per_row):
                    row_targets = geo_targets[row_start:row_start + cols_per_row]
                    cols = st.columns(len(row_targets))
                    for ci, tgt in enumerate(row_targets):
                        label = TARGET_DISPLAY.get(tgt, tgt)
                        with cols[ci]:
                            fig = px.scatter_mapbox(
                                map_data, lat="latitude", lon="longitude",
                                hover_name="location_name",
                                hover_data={
                                    "country": True, label: ":.3f",
                                    "latitude": ":.2f", "longitude": ":.2f",
                                },
                                color=label,
                                size=np.abs(map_data[label]).clip(lower=0.1).tolist(),
                                color_continuous_scale="Turbo",
                                mapbox_style="carto-positron",
                                zoom=1, height=400,
                                title=label,
                            )
                            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig, use_container_width=True)

        # ── Country Bar Chart — grouped by target ────────────────
        with tab_bar:
            country_agg = map_data.groupby("country")[agg_cols].mean().reset_index()
            melted = country_agg.melt(
                id_vars="country", var_name="Target", value_name="Value",
            )
            fig = px.bar(
                melted, x="country", y="Value", color="Target",
                barmode="group",
                title="Mean Forecast by Country",
                labels={"country": "Country"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Per-target box plots showing spread across locations
            if len(geo_targets) > 1:
                st.subheader("Distribution across locations")
                melted_loc = map_data.melt(
                    id_vars=["location_name", "country"],
                    value_vars=agg_cols,
                    var_name="Target", value_name="Value",
                )
                fig_box = px.box(
                    melted_loc, x="Target", y="Value", color="Target",
                    points="all",
                    title="Forecast distribution by target across all locations",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_box.update_layout(showlegend=False, template="plotly_white")
                st.plotly_chart(fig_box, use_container_width=True)

        # ── Radar chart — multi-target per country ───────────────
        with tab_radar:
            if len(geo_targets) < 2:
                st.info("Select at least 2 targets to see the radar chart.")
            else:
                country_agg = map_data.groupby("country")[agg_cols].mean().reset_index()
                # Normalise to [0, 1] for comparability on radar
                normalised = country_agg[agg_cols].copy()
                for col in agg_cols:
                    col_min, col_max = normalised[col].min(), normalised[col].max()
                    if col_max > col_min:
                        normalised[col] = (normalised[col] - col_min) / (col_max - col_min)
                    else:
                        normalised[col] = 0.5

                fig = go.Figure()
                colors = px.colors.qualitative.Vivid
                for i, (_, row) in enumerate(country_agg.iterrows()):
                    country_name = row["country"]
                    vals = normalised.iloc[i].tolist()
                    fig.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],  # close the polygon
                        theta=agg_cols + [agg_cols[0]],
                        fill="toself",
                        name=country_name,
                        line=dict(color=colors[i % len(colors)]),
                        opacity=0.6,
                    ))
                fig.update_layout(
                    title="Normalised Target Profiles by Country",
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    template="plotly_white",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Values are min-max normalised per target for visual comparison. "
                           "Hover for country details.")

        # ── Data Table ───────────────────────────────────────────
        with tab_table:
            display_cols = ["location_name", "country", "latitude", "longitude"] + agg_cols
            st.dataframe(
                map_data[display_cols].style.format(
                    {c: "{:.4f}" for c in agg_cols + ["latitude", "longitude"]}
                ),
                use_container_width=True,
            )
            csv = map_data[display_cols].to_csv(index=False)
            st.download_button("Download CSV", csv, "geographic_forecast.csv", "text/csv")

    # =================================================================
    # PAGE 5 — Model Explorer
    # =================================================================

    def page_model_explorer(self):
        st.header("\U0001F9E0 Model Explorer")
        st.markdown("Compare trained model architectures, test metrics, and training history.")

        available = _get_available_models(self.artifacts_dir)
        if not available:
            st.warning("No trained models found in artifacts/models/.")
            return

        # ── Load all metadata ────────────────────────────────────
        all_meta: Dict[str, dict] = {}
        for name, path in available.items():
            all_meta[name] = _load_model_metadata(path)

        tab_summary, tab_per_target, tab_curves = st.tabs(
            ["Summary Comparison", "Per-Target Breakdown", "Training Curves"]
        )

        # ── Summary Table ────────────────────────────────────────
        with tab_summary:
            rows = []
            for name, meta in all_meta.items():
                rows.append({
                    "Model": MODEL_NAMES.get(name, name),
                    "Total Params": meta.get("total_params", "N/A"),
                    "Avg R\u00b2": meta.get("avg_r2", None),
                    "Avg MAE": meta.get("avg_mae", None),
                    "Avg RMSE": meta.get("avg_rmse", None),
                    "Train Time (s)": meta.get("train_time_s", None),
                    "Seq Length": meta.get("seq_len", None),
                })
            summary_df = pd.DataFrame(rows)

            # Highlight best R²
            st.dataframe(
                summary_df.style.format(
                    {"Avg R\u00b2": "{:.4f}", "Avg MAE": "{:.4f}",
                     "Avg RMSE": "{:.4f}", "Train Time (s)": "{:.1f}"},
                    na_rep="—",
                ).highlight_max(subset=["Avg R\u00b2"], color="#2ecc71")
                 .highlight_min(subset=["Avg MAE", "Avg RMSE"], color="#2ecc71"),
                use_container_width=True,
            )

            # Bar chart of avg R²
            if any(r.get("Avg R\u00b2") is not None for r in rows):
                fig = px.bar(
                    summary_df.dropna(subset=["Avg R\u00b2"]),
                    x="Model", y="Avg R\u00b2",
                    color="Model",
                    title="Model Comparison — Average R\u00b2",
                    text_auto=".4f",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # ── Per-Target Breakdown ─────────────────────────────────
        with tab_per_target:
            rows_pt = []
            for name, meta in all_meta.items():
                per_target = meta.get("per_target", {})
                for tgt, vals in per_target.items():
                    if isinstance(vals, dict):
                        rows_pt.append({
                            "Model": MODEL_NAMES.get(name, name),
                            "Target": TARGET_DISPLAY.get(tgt, tgt),
                            "R\u00b2": vals.get("R2"),
                            "MAE": vals.get("MAE"),
                            "RMSE": vals.get("RMSE"),
                        })

            if rows_pt:
                pt_df = pd.DataFrame(rows_pt)

                # R² grouped bar chart
                fig = px.bar(
                    pt_df, x="Target", y="R\u00b2", color="Model",
                    barmode="group",
                    title="R\u00b2 by Target and Model",
                    text_auto=".3f",
                )
                fig.update_layout(xaxis_tickangle=-30, height=500)
                st.plotly_chart(fig, use_container_width=True)

                # MAE grouped bar chart
                fig2 = px.bar(
                    pt_df, x="Target", y="MAE", color="Model",
                    barmode="group",
                    title="MAE by Target and Model",
                    text_auto=".2f",
                )
                fig2.update_layout(xaxis_tickangle=-30, height=500)
                st.plotly_chart(fig2, use_container_width=True)

                # Full table
                with st.expander("Full per-target metrics table"):
                    st.dataframe(
                        pt_df.style.format(
                            {"R\u00b2": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"},
                            na_rep="—",
                        ),
                        use_container_width=True,
                    )
            else:
                st.info("No per-target metrics found in model metadata.")

        # ── Training Curves ──────────────────────────────────────
        with tab_curves:
            st.markdown("Training loss curves from model metadata (if available).")

            # Check if any model has training history saved as artefact
            has_history = False
            for name, meta in all_meta.items():
                epochs_trained = meta.get("epochs_trained") or meta.get("seq_len")
                if epochs_trained:
                    has_history = True
                    break

            if has_history:
                # Show available metrics from metadata
                for name, meta in all_meta.items():
                    display_name = MODEL_NAMES.get(name, name)
                    final_train = meta.get("final_train_loss")
                    final_val = meta.get("final_val_loss")
                    epochs_trained = meta.get("epochs_trained")

                    if epochs_trained:
                        st.subheader(display_name)
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Epochs Trained", epochs_trained)
                        if final_train is not None:
                            mc2.metric("Final Train Loss", f"{final_train:.6f}")
                        if final_val is not None:
                            mc3.metric("Final Val Loss", f"{final_val:.6f}")

                        # Show avg metrics as a bar
                        avg_r2 = meta.get("avg_r2")
                        avg_mae = meta.get("avg_mae")
                        if avg_r2 is not None:
                            st.progress(
                                min(max(avg_r2, 0.0), 1.0),
                                text=f"Average R\u00b2: {avg_r2:.4f}",
                            )
            else:
                st.info(
                    "No training history found. The training pipeline saves "
                    "epoch-level metrics to MLflow / W&B."
                )


# =====================================================================
# App entry-point
# =====================================================================

def main():
    """Launch the Streamlit dashboard."""
    app = StreamlitFrontend()
    app.run()


if __name__ == "__main__":
    main()       