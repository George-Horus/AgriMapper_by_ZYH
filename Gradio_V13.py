"""
AgriMapper Gradio V13
English-only portrait-style interface for thesis screenshots.
"""

import ast
import io
import os

import matplotlib as mpl
mpl.use("Agg")
import geopandas as gpd
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

from S1_preprocessing import d_data_process_main
from S2_model_training import model_train_main
from S3_prediction_vis import Estimation_map, calculate_indices_map, grid_grade_diagram


Image.MAX_IMAGE_PIXELS = None

BAND_CHOICES = ["BLUE", "GREEN", "NIR", "RED", "RedEdge", "other"]
INDEX_OPTIONS = [
    "ARI", "ARVI", "BDRVI", "CCCI", "CIG", "CIRE", "CIVE", "CRI700", "CVI", "DATT", "DVI", "EBVI",
    "EGVI", "ERVI", "EVI1", "EVI2", "ExG", "ExGR", "EXR1", "EXR2", "GARI", "GDVI1", "GDVI2", "GEMI",
    "GLI1", "GLI2", "GNDVI", "GOSAVI", "GRDVI", "GRVI1", "GRVI2", "GSAVI", "GWDRVI", "MCARI",
    "MCARI1", "MCARI2", "MCARI3", "MCARI4", "MDD", "MEVI", "MGRVI", "MNDI1", "MNDI2", "MNDRE1",
    "MNDRE2", "MNDVI", "MNLI", "MNSI", "MRETVI", "MSAVI", "MSR", "MSR_G", "MSR_R", "MSR_RE", "MSRRE",
    "MSRREDRE", "MTCARI", "MTCI", "NB", "NDGI", "NDRE", "NDVI", "NDWI", "NG", "NGBDI", "NGI",
    "NGRDI", "NNIR", "NNRI", "NPCI", "NR", "NREI1", "NREI2", "NRI", "NROV", "OSAVI1", "OSAVI2",
    "PNDVI", "PPR", "PRI", "PSRI1", "PSRI2", "PVR", "RBI", "RBNDVI", "RDVI1", "RDVI2", "REDVI",
    "REGDVI", "REGNDVI", "REGRVI", "RENDVI", "REOSAVI", "RERVI", "RESAVI", "RESR", "RETVI",
    "REWDRVI", "RGBVI", "RGI", "RI", "RTVIcore", "RVI", "SAVI", "SIPI1", "SIPI2", "SRI", "SRPI",
    "TCARI", "TVI1", "TVI2", "VARI", "VI700", "VIopt", "WBI", "WDRVI", "WI"
]
COLORMAP_OPTIONS = sorted(
    [
        "viridis", "plasma", "inferno", "magma", "cividis",
        "RdYlGn", "RdBu", "coolwarm", "bwr",
        "YlGn", "terrain", "nipy_spectral", "twilight", "hsv"
    ]
)

APP_CSS = """
:root {
  --bg: #f3efe7;
  --panel: #fffdf8;
  --panel-soft: #f8f5ee;
  --border: #2e4359;
  --border-soft: #8ca0b3;
  --accent: #a65e2e;
  --accent-2: #23415e;
  --text: #1e2b36;
  --muted: #586774;
  --shadow: 0 16px 34px rgba(28, 45, 62, 0.10);
}

body {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.26), rgba(255, 255, 255, 0.06)),
    linear-gradient(180deg, #f7f3eb 0%, #eef2f5 100%);
}

.gradio-container {
  max-width: 1380px !important;
  padding: 18px 18px 36px !important;
  color: var(--text);
  font-family: "Segoe UI", "Aptos", sans-serif;
}

#app-shell {
  gap: 18px;
}

.hero-panel {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(249, 246, 239, 0.98));
  border: 3px solid var(--border);
  border-radius: 22px;
  box-shadow: var(--shadow);
  padding: 18px 24px;
}

.hero-title {
  margin: 0;
  color: var(--text);
  font-family: "Georgia", "Times New Roman", serif;
  font-size: 34px;
  line-height: 1.12;
  font-weight: 700;
}

.section-heading {
  margin-bottom: 8px;
}

.section-tag {
  display: inline-block;
  padding: 7px 14px;
  border-radius: 999px;
  border: 2px solid var(--border);
  background: rgba(166, 94, 46, 0.08);
  color: var(--accent);
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.section-heading h2 {
  margin: 10px 0 0;
  color: var(--text);
  font-family: "Georgia", "Times New Roman", serif;
  font-size: 28px;
  line-height: 1.2;
  font-weight: 700;
}

.section-heading p {
  display: none;
}

.frame-card {
  background: var(--panel);
  border: 2.5px solid var(--border);
  border-radius: 24px;
  padding: 16px !important;
  box-shadow: var(--shadow);
}

.frame-soft {
  background: var(--panel-soft);
}

.frame-output {
  background: linear-gradient(180deg, #fffdf8 0%, #f6f9fc 100%);
}

.box-header {
  margin-bottom: 10px;
}

.box-kicker {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: var(--accent-2);
  color: white;
  font-size: 14px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.box-title {
  margin-top: 10px;
  color: var(--text);
  font-family: "Georgia", "Times New Roman", serif;
  font-size: 24px;
  line-height: 1.2;
  font-weight: 700;
}

.box-desc {
  display: none;
}

.note-list {
  margin: 0;
  padding-left: 22px;
}

.note-list li {
  color: var(--muted);
  font-size: 17px;
  line-height: 1.8;
  margin-bottom: 6px;
}

.gradio-container .tab-nav button {
  font-size: 17px !important;
  font-weight: 800 !important;
  padding: 12px 14px !important;
  border-radius: 16px 16px 0 0 !important;
}

.gradio-container h3 {
  font-size: 22px !important;
}

.gradio-container label,
.gradio-container .prose,
.gradio-container .form label,
.gradio-container .block-info,
.gradio-container .wrap {
  font-size: 16px !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  font-size: 16px !important;
  min-height: 46px !important;
  background: rgba(255, 255, 255, 0.96) !important;
}

.gradio-container textarea {
  line-height: 1.7 !important;
}

#dataset-picker fieldset,
.choice-state fieldset {
  display: grid;
  gap: 10px;
  border: 0;
  padding: 0 !important;
  margin: 0 !important;
}

#dataset-picker fieldset > div,
.choice-state fieldset > div {
  display: grid !important;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
  gap: 8px;
}

#dataset-picker fieldset label,
.choice-state fieldset label {
  display: flex !important;
  align-items: center;
  gap: 10px;
  min-height: 48px;
  padding: 8px 12px;
  border: 2px solid #c4ced5;
  border-radius: 8px;
  background: #ffffff;
  color: var(--text);
  cursor: pointer;
  transition: border-color 140ms ease, background-color 140ms ease, box-shadow 140ms ease;
}

#dataset-picker fieldset label:hover,
.choice-state fieldset label:hover {
  border-color: #637f8f;
  background: #f4f7f8;
}

#dataset-picker fieldset label:has(input:checked),
.choice-state fieldset label:has(input:checked) {
  border-color: #176b4b;
  background: #dff2e9;
  box-shadow: inset 4px 0 0 #176b4b;
}

#dataset-picker fieldset label::after,
.choice-state fieldset label::after {
  content: "Not selected";
  flex: 0 0 auto;
  margin-left: auto;
  padding: 3px 7px;
  border-radius: 999px;
  background: #edf0f2;
  color: #4d5960;
  font-size: 11px;
  font-weight: 700;
  line-height: 1.2;
  white-space: nowrap;
}

#dataset-picker fieldset label:has(input:checked)::after,
.choice-state fieldset label:has(input:checked)::after {
  content: "Selected";
  background: #176b4b;
  color: #ffffff;
}

#dataset-picker fieldset label:has(input:checked) span,
.choice-state fieldset label:has(input:checked) span {
  color: #123d2d;
  font-weight: 800;
}

#dataset-picker input[type="checkbox"],
.choice-state input[type="checkbox"],
.choice-state input[type="radio"] {
  appearance: none;
  display: grid !important;
  flex: 0 0 20px;
  place-content: center;
  width: 20px !important;
  min-width: 20px;
  height: 20px !important;
  min-height: 20px !important;
  margin: 0;
  padding: 0 !important;
  border: 2px solid #83939d;
  border-radius: 4px;
  background: #ffffff !important;
  cursor: pointer;
}

#dataset-picker input[type="checkbox"],
.choice-state input[type="checkbox"] {
  border-radius: 4px;
}

.choice-state input[type="radio"] {
  border-radius: 50%;
}

#dataset-picker input[type="checkbox"]::after,
.choice-state input[type="checkbox"]::after {
  content: "";
  width: 5px;
  height: 10px;
  border: solid #ffffff;
  border-width: 0 2.5px 2.5px 0;
  opacity: 0;
  transform: rotate(45deg) translate(-1px, -1px);
}

#dataset-picker input[type="checkbox"]:checked,
.choice-state input[type="checkbox"]:checked,
.choice-state input[type="radio"]:checked {
  border-color: #176b4b;
  background: #176b4b !important;
}

#dataset-picker input[type="checkbox"]:checked::after,
.choice-state input[type="checkbox"]:checked::after {
  opacity: 1;
}

.choice-state input[type="radio"]::after {
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ffffff;
  opacity: 0;
}

.choice-state input[type="radio"]:checked::after {
  opacity: 1;
}

#dataset-picker fieldset label span,
.choice-state fieldset label span {
  line-height: 1.25;
}

#dataset-picker input[type="checkbox"]:focus-visible,
.choice-state input[type="checkbox"]:focus-visible,
.choice-state input[type="radio"]:focus-visible {
  outline: 3px solid rgba(23, 107, 75, 0.32);
  outline-offset: 2px;
}

.gradio-container [role="option"][aria-selected="true"] {
  background: #dff2e9 !important;
  box-shadow: inset 4px 0 0 #176b4b;
  color: #123d2d !important;
  font-weight: 800 !important;
}

.gradio-container [role="option"][aria-selected="true"]::after {
  content: "Selected";
  float: right;
  margin-left: 12px;
  padding: 2px 7px;
  border-radius: 999px;
  background: #176b4b;
  color: #ffffff;
  font-size: 11px;
  line-height: 1.3;
}

.gradio-container button {
  font-size: 17px !important;
  font-weight: 800 !important;
  min-height: 48px !important;
  border-radius: 14px !important;
}

.gradio-container table,
.gradio-container th,
.gradio-container td {
  font-size: 17px !important;
}

.path-box textarea,
.path-box input {
  font-family: "Consolas", "Courier New", monospace !important;
  font-size: 16px !important;
}

.portrait-gallery {
  min-height: 220px;
}

.portrait-gallery img,
.portrait-image img {
  border: 2px solid var(--border-soft) !important;
  border-radius: 18px !important;
}

@media (max-width: 900px) {
  .hero-title {
    font-size: 28px;
  }

  .section-heading h2 {
    font-size: 24px;
  }

  .box-title {
    font-size: 21px;
  }
}
"""


def normalize_file_input(file_input):
    if file_input is None:
        return []
    if isinstance(file_input, (list, tuple)):
        return [path for item in file_input if (path := extract_path(item))]
    path = extract_path(file_input)
    return [path] if path else []


def extract_path(file_item):
    if file_item is None:
        return None
    if isinstance(file_item, str):
        return file_item
    if isinstance(file_item, dict):
        return file_item.get("name") or file_item.get("path")
    return getattr(file_item, "name", None)


def resolve_output_dir(output_dir):
    if isinstance(output_dir, str) and output_dir.strip():
        return output_dir.strip()
    return "output"


def safe_literal_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str) and not value.strip():
        return []
    return ast.literal_eval(value)


def hidden_dropdown_update():
    return gr.update(choices=[], value=None, visible=False, interactive=False)


def render_hero():
    return """
    <div class="hero-panel">
      <div class="hero-title">Crop Phenotyping and Nitrogen Estimation Platform Based on Multispectral Remote Sensing</div>
    </div>
    """


def render_section_header(step_tag, title, description):
    return f"""
    <div class="section-heading">
      <div class="section-tag">{step_tag}</div>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
    """


def render_box_header(box_tag, title, description):
    return f"""
    <div class="box-header">
      <div class="box-kicker">{box_tag}</div>
      <div class="box-title">{title}</div>
      <div class="box-desc">{description}</div>
    </div>
    """


def generate_dropdowns(tif_input):
    tif_list = normalize_file_input(tif_input)
    if not tif_list:
        return [hidden_dropdown_update() for _ in range(10)]

    try:
        with rasterio.open(tif_list[0]) as src:
            band_count = src.count
    except Exception:
        return [hidden_dropdown_update() for _ in range(10)]

    dropdowns = []
    for index in range(min(band_count, 10)):
        dropdowns.append(
            gr.update(
                choices=BAND_CHOICES,
                value="other",
                label=f"Band {index + 1}",
                visible=True,
                interactive=True,
            )
        )

    while len(dropdowns) < 10:
        dropdowns.append(hidden_dropdown_update())
    return dropdowns


def generate_checkbox(tif_input):
    tif_list = normalize_file_input(tif_input)
    if not tif_list:
        return gr.update(choices=[], value=[], visible=False, interactive=False)

    dataset_names = [f"Dataset {index + 1}" for index in range(len(tif_list))]
    return gr.update(
        choices=dataset_names,
        value=dataset_names,
        visible=True,
        interactive=True,
    )


def generate_number(file_input):
    file_paths = normalize_file_input(file_input)
    if not file_paths:
        return gr.update(
            visible=True,
            value=1,
            label="Sampling Window Size",
        )

    file_path = file_paths[0]
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".shp":
        return gr.update(
            visible=True,
            value=1,
            label="Sampling Window Size",
        )

    try:
        gdf = gpd.read_file(file_path)
        if all(geom_type == "Point" for geom_type in gdf.geom_type):
            return gr.update(
                visible=True,
                value=1,
                label="Sampling Window Size",
            )
        return gr.update(visible=False, value=None)
    except Exception:
        return gr.update(visible=False, value=None)


def generate_components(tif_input, file_input):
    number_update = generate_number(file_input)
    dropdown_updates = generate_dropdowns(tif_input)
    checkbox_update = generate_checkbox(tif_input)
    return [number_update] + dropdown_updates + [checkbox_update]


def STEP1_data_process_main(tif_paths, coord_paths, output_dir, target_column, window_size, *args):
    tif_paths = normalize_file_input(tif_paths)
    coord_paths = normalize_file_input(coord_paths)
    output_dir = resolve_output_dir(output_dir)

    if not tif_paths:
        raise ValueError("Please upload at least one multispectral image.")
    if not coord_paths:
        raise ValueError("Please upload a coordinate or sampling file.")
    if not target_column:
        raise ValueError("Please enter the target column name.")

    dropdown_values = args[:-1]
    checkbox_values = args[-1] or []

    selected_dataset_indices = []
    for item in checkbox_values:
        try:
            selected_dataset_indices.append(int("".join(filter(str.isdigit, str(item)))) - 1)
        except (TypeError, ValueError):
            continue

    band_match = [value for value in dropdown_values if value is not None]

    df_features, feature_list, _, result_dir = d_data_process_main.data_process_main(
        tif_paths,
        coord_paths,
        band_match,
        target_column,
        output_dir,
        selected_dataset_indices,
        window_size,
    )

    image_paths = sorted(
        [
            os.path.join(result_dir, file_name)
            for file_name in os.listdir(result_dir)
            if file_name.lower().endswith(".png")
        ]
    )
    preview_images = [Image.open(path) for path in image_paths]

    return (
        gr.update(value=df_features, visible=True),
        gr.update(value=str(feature_list), visible=True),
        gr.update(value=preview_images, visible=True),
        gr.update(value=str(band_match), visible=True),
    )


def model_data_upload(option):
    return gr.update(visible=(option == "file_upload"))


def STEP2_model_train_main(
    option,
    file_input,
    best_features_df,
    output_dir_input,
    validation_mode="standard",
    cv_model_group="all",
):
    output_dir_input = resolve_output_dir(output_dir_input)

    if option == "file_upload":
        upload_paths = normalize_file_input(file_input)
        if not upload_paths:
            raise ValueError("Please upload a .csv or .xlsx modeling file.")
        file_path = upload_paths[0]
        if file_path.endswith(".csv"):
            best_features_df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            best_features_df = pd.read_excel(file_path)
        else:
            raise ValueError("Only .csv and .xlsx files are supported.")
    elif best_features_df is None or len(best_features_df) == 0:
        raise ValueError("Please finish feature engineering first, or switch to file upload mode.")

    if validation_mode == "cross_validation":
        # S1 has already produced the fixed feature table.  CV belongs only
        # to this S2 model-fitting branch; no Pearson/RFE step is repeated.
        model_performance_df, figures, model_path, save_path = model_train_main.Cross_validation_main(
            best_features_df,
            output_dir_input,
            requested_models=(cv_model_group or "all",),
            n_splits=5,
        )
        image_files = [
            str(path)
            for path in figures.values()
            if path and os.path.exists(path)
        ]
        images = [Image.open(path) for path in image_files]
        return (
            gr.update(value=model_performance_df),
            gr.update(value=images),
            # Cross-validation writes fold evaluation artifacts. It does not
            # create one deployable ML/DL/Auto model for the prediction tab.
            gr.update(value=""),
        )

    model_performance_df, _, model_path, save_path = model_train_main.Training_main(
        best_features_df,
        output_dir_input,
    )

    sort_order = ["ML", "DL", "auto", "best", "model"]
    image_files = [
        os.path.join(save_path, file_name)
        for file_name in os.listdir(save_path)
        if file_name.lower().endswith(".png")
    ]

    def custom_sort(file_path):
        filename = os.path.basename(file_path)
        for sort_index, prefix in enumerate(sort_order):
            if filename.startswith(prefix):
                return sort_index
        return len(sort_order)

    sorted_files = sorted(image_files, key=custom_sort)
    images = [Image.open(path) for path in sorted_files]

    return (
        gr.update(value=model_performance_df),
        gr.update(value=images),
        gr.update(value=str(model_path)),
    )


def STEP3_Estimation_map_main(multi_tif_data_path, model_path, model_selection, bandlist, index_list, output_dir_input):
    tif_paths = normalize_file_input(multi_tif_data_path)
    if not tif_paths:
        raise ValueError("Please upload at least one TIF file for prediction.")

    model_path = safe_literal_list(model_path)
    if len(model_path) < 3:
        raise ValueError(
            "The prediction tab requires the three model paths returned by standard training. "
            "Five-fold cross-validation outputs evaluation artifacts only; run standard training before prediction."
        )
    bandlist = safe_literal_list(bandlist)
    index_list = safe_literal_list(index_list)
    output_dir_input = resolve_output_dir(output_dir_input)

    images, tif_paths = Estimation_map.Estimation_map_main(
        tif_paths,
        model_path,
        model_selection,
        bandlist,
        index_list,
        output_dir_input,
    )

    return gr.update(value=images), gr.update(value="\n".join(tif_paths))


def STEP4_indices_map_main(multi_tif_data_path, bandlist, selected_option, output_dir_input):
    tif_paths = normalize_file_input(multi_tif_data_path)
    if not tif_paths:
        raise ValueError("Please upload a TIF file for vegetation index calculation.")
    if not selected_option:
        raise ValueError("Please select one vegetation index.")

    bandlist = safe_literal_list(bandlist)
    output_dir_input = resolve_output_dir(output_dir_input)

    tif_path, hist_img = calculate_indices_map.indices_map_main(
        tif_paths[0],
        bandlist,
        [selected_option],
        output_dir_input,
    )

    return gr.update(value=tif_path), gr.update(value=hist_img)


def STEP5_visualize_tif(tif_path, colormap="RdYlGn", clip_percentile=5):
    tif_paths = normalize_file_input(tif_path)
    if not tif_paths:
        raise ValueError("Please upload a GeoTIFF file for visualization.")

    with rasterio.open(tif_paths[0]) as src:
        data = src.read(1).astype(np.float32)

    valid = data[~np.isnan(data)]
    if valid.size == 0:
        raise ValueError("No valid pixels are available for visualization.")

    lower = np.percentile(valid, clip_percentile)
    upper = np.percentile(valid, 100 - clip_percentile)
    data_clipped = np.clip(data, lower, upper)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.2, 6.6), dpi=180)
    fig.patch.set_facecolor("#fffdf8")
    ax.set_facecolor("#fffdf8")
    norm = mpl.colors.Normalize(vmin=np.nanmin(data_clipped), vmax=np.nanmax(data_clipped))
    image = ax.imshow(data_clipped, cmap=colormap, norm=norm)
    ax.axis("off")
    colorbar = plt.colorbar(image, ax=ax, fraction=0.042, pad=0.03)
    colorbar.set_label("Prediction Value", fontsize=15)
    colorbar.ax.tick_params(labelsize=13)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    buffer.seek(0)
    preview = Image.open(buffer).copy()
    buffer.close()
    return preview


def STEP6_grid_main(tif_file, grid_width_meters, N_std, B_std, output_dir_input):
    tif_paths = normalize_file_input(tif_file)
    if not tif_paths:
        raise ValueError("Please upload a nitrogen prediction TIF file.")

    output_dir_input = resolve_output_dir(output_dir_input)
    fig, save_path = grid_grade_diagram.grid_main(
        tif_paths[0],
        grid_width_meters,
        N_std,
        B_std,
        output_dir_input,
    )
    return fig, save_path


def create_data_input_tab():
    with gr.Tab("01 Data Input"):
        gr.HTML(
            render_section_header(
                "Stage 01",
                "Project Input and Basic Configuration",
                "Compact input layout for project setup.",
            )
        )

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group(elem_classes=["frame-card", "frame-output"]):
                    gr.HTML(
                        render_box_header(
                            "Input Files",
                            "Multispectral imagery and field data",
                            "",
                        )
                    )
                    tif_input = gr.File(
                        type="filepath",
                        label="Upload multispectral image files (.tif)",
                        file_count="multiple",
                    )
                    file_input = gr.File(
                        type="filepath",
                        label="Upload location or sampling files (.csv, .xlsx, .shp, .txt)",
                        file_count="multiple",
                    )
            with gr.Column(scale=5):
                with gr.Group(elem_classes="frame-card"):
                    gr.HTML(
                        render_box_header(
                            "Project Setup",
                            "Output and target settings",
                            "",
                        )
                    )
                    output_dir = gr.Textbox(
                        label="Output directory",
                        value="output",
                        placeholder="Example: output or D:/Agrimapper/output",
                        elem_classes="path-box",
                    )
                    target_value = gr.Textbox(
                        label="Target column name",
                        placeholder="Example: Nitrogen_Content",
                    )
                    upload_btn = gr.Button("Read files and generate the parameter panel", variant="primary")

    return {
        "tif_input": tif_input,
        "file_input": file_input,
        "output_dir": output_dir,
        "target_value": target_value,
        "upload_btn": upload_btn,
    }


def create_feature_engineering_tab():
    with gr.Tab("02 Feature Engineering"):
        gr.HTML(
            render_section_header(
                "Stage 02",
                "Band Mapping, Sampling Settings, and Feature Filtering",
                "Compact feature-engineering layout.",
            )
        )

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group(elem_classes="frame-card"):
                    gr.HTML(
                        render_box_header(
                            "Band Mapping",
                            "Band-to-channel assignment",
                            "",
                        )
                    )
                    with gr.Row():
                        first_row = []
                        for index in range(5):
                            first_row.append(gr.Dropdown(visible=False, interactive=False, label=f"Band {index + 1}"))
                    with gr.Row():
                        second_row = []
                        for index in range(5, 10):
                            second_row.append(gr.Dropdown(visible=False, interactive=False, label=f"Band {index + 1}"))
                    dropdowns = first_row + second_row
            with gr.Column(scale=5):
                with gr.Group(elem_classes="frame-card"):
                    gr.HTML(
                        render_box_header(
                            "Settings",
                            "Sampling and dataset selection",
                            "",
                        )
                    )
                    window_size_input = gr.Number(
                        label="Sampling Window Size",
                        value=1,
                        precision=0,
                        visible=False,
                    )
                    dataset_checkbox = gr.CheckboxGroup(
                        label="Datasets included in feature filtering",
                        visible=False,
                        elem_id="dataset-picker",
                        elem_classes="choice-state",
                    )
                    submit_btn = gr.Button("Run feature filtering and export results", variant="primary")

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Group(elem_classes=["frame-card", "frame-output"]):
                    gr.HTML(
                        render_box_header(
                            "Output Summary",
                            "Filtered features and band mapping",
                            "",
                        )
                    )
                    sorted_features = gr.Textbox(
                        label="Filtered feature list",
                        lines=7,
                        interactive=False,
                        visible=False,
                        elem_classes="path-box",
                    )
                    band_order = gr.Textbox(
                        label="Band mapping result",
                        lines=4,
                        interactive=False,
                        visible=False,
                        elem_classes="path-box",
                    )
            with gr.Column(scale=7):
                with gr.Group(elem_classes=["frame-card", "frame-output"]):
                    gr.HTML(
                        render_box_header(
                            "Feature Table",
                            "Optimal feature preview",
                            "",
                        )
                    )
                    features_df = gr.Dataframe(
                        label="Optimal Feature Preview",
                        interactive=True,
                        max_height=520,
                        show_search="search",
                        visible=False,
                    )

        with gr.Group(elem_classes=["frame-card", "frame-output"]):
            gr.HTML(
                render_box_header(
                    "Feature Figures",
                    "Feature selection visualizations",
                    "",
                )
            )
            gallery = gr.Gallery(
                label="Feature Selection Figures",
                columns=2,
                format="png",
                visible=False,
                elem_classes="portrait-gallery",
            )

    return {
        "dropdowns": dropdowns,
        "window_size_input": window_size_input,
        "dataset_checkbox": dataset_checkbox,
        "submit_btn": submit_btn,
        "sorted_features": sorted_features,
        "band_order": band_order,
        "features_df": features_df,
        "gallery": gallery,
    }


def create_model_training_tab():
    with gr.Tab("03 Model Training"):
        gr.HTML(
            render_section_header(
                "Stage 03",
                "Regression Model Training and Comparison",
                "Compact model-training layout.",
            )
        )

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group(elem_classes="frame-card"):
                    gr.HTML(
                        render_box_header(
                            "Training Mode",
                            "Choose the modeling data source",
                            "",
                        )
                    )
                    model_option = gr.Radio(
                        choices=[
                            ("Upload a modeling file", "file_upload"),
                            ("Use the S1 fixed-feature table", "normal_mode"),
                        ],
                        label="Modeling data source",
                        value="file_upload",
                        elem_classes="choice-state",
                    )
                    validation_mode = gr.Radio(
                        choices=[
                            ("Standard train/test comparison", "standard"),
                            ("Five-fold cross-validation (modelling only)", "cross_validation"),
                        ],
                        label="Validation workflow",
                        value="standard",
                        elem_classes="choice-state",
                    )
                    cv_model_group = gr.Dropdown(
                        choices=["all", "ml", "dl", "automl"],
                        value="all",
                        label="Cross-validation model group",
                    )
                    file_upload = gr.File(
                        label="Upload S1 fixed-feature table (.xlsx or .csv)",
                        type="filepath",
                        file_count="multiple",
                    )
                    submit_btn = gr.Button("Train models and generate comparison outputs", variant="primary")
                    model_path = gr.Textbox(
                        label="Best model save path",
                        lines=4,
                        interactive=False,
                        elem_classes="path-box",
                    )
            with gr.Column(scale=8):
                with gr.Group(elem_classes=["frame-card", "frame-output"]):
                    gr.HTML(
                        render_box_header(
                            "Performance Table",
                            "Model performance ranking",
                            "",
                        )
                    )
                    performance_df = gr.Dataframe(
                        label="Model Performance Ranking",
                        interactive=True,
                        max_height=430,
                        show_search="search",
                    )

        with gr.Group(elem_classes=["frame-card", "frame-output"]):
            gr.HTML(
                render_box_header(
                    "Model Figures",
                    "Training and comparison visualizations",
                    "",
                )
            )
            gallery = gr.Gallery(
                label="Model Comparison Figures",
                columns=2,
                format="png",
                elem_classes="portrait-gallery",
            )

    return {
        "model_option": model_option,
        "validation_mode": validation_mode,
        "cv_model_group": cv_model_group,
        "file_upload": file_upload,
        "submit_btn": submit_btn,
        "performance_df": performance_df,
        "gallery": gallery,
        "model_path": model_path,
    }


def create_estimation_tab():
    with gr.Tab("04 Estimation Results"):
        gr.HTML(
            render_section_header(
                "Stage 04",
                "Prediction Map Generation and Vegetation Index Mapping",
                "Compact estimation layout.",
            )
        )

        with gr.Row():
            with gr.Column(scale=6):
                with gr.Group(elem_classes="frame-card"):
                    gr.HTML(
                        render_box_header(
                            "Prediction Maps",
                            "Generate nitrogen estimation outputs",
                            "",
                        )
                    )
                    model_option = gr.Radio(
                        choices=[
                            ("Best machine learning model", "ML"),
                            ("Best deep learning model", "DL"),
                            ("Best ensemble model", "Auto"),
                        ],
                        label="Prediction model selection",
                        value="Auto",
                        elem_classes="choice-state",
                    )
                    multi_tif_path = gr.File(
                        label="Upload TIF files for prediction",
                        type="filepath",
                        file_count="multiple",
                    )
                    submit_btn = gr.Button("Generate prediction maps and summary figures", variant="primary")
                    output_txt = gr.Textbox(
                        label="Prediction map output path",
                        lines=4,
                        interactive=False,
                        elem_classes="path-box",
                    )
                    gallery = gr.Gallery(
                        label="Prediction Histograms",
                        columns=2,
                        format="png",
                        elem_classes="portrait-gallery",
                    )
            with gr.Column(scale=6):
                with gr.Group(elem_classes=["frame-card", "frame-output"]):
                    gr.HTML(
                        render_box_header(
                            "Vegetation Index Maps",
                            "Generate a single index map and histogram",
                            "",
                        )
                    )
                    index_dropdown = gr.Dropdown(
                        choices=INDEX_OPTIONS,
                        label="Select one vegetation index",
                        interactive=True,
                    )
                    index_tif_path = gr.File(
                        label="Upload a TIF file for index calculation",
                        type="filepath",
                        file_count="multiple",
                    )
                    index_btn = gr.Button("Generate index map and grayscale histogram", variant="primary")
                    output_index_txt = gr.Textbox(
                        label="Index map output path",
                        interactive=False,
                        elem_classes="path-box",
                    )
                    histogram = gr.Image(
                        label="Grayscale histogram",
                        type="pil",
                        format="png",
                        elem_classes="portrait-image",
                    )

    return {
        "model_option": model_option,
        "multi_tif_path": multi_tif_path,
        "submit_btn": submit_btn,
        "output_txt": output_txt,
        "gallery": gallery,
        "index_dropdown": index_dropdown,
        "index_tif_path": index_tif_path,
        "index_btn": index_btn,
        "output_index_txt": output_index_txt,
        "histogram": histogram,
    }


def create_prescription_tab():
    with gr.Tab("05 Prescription Mapping"):
        gr.HTML(
            render_section_header(
                "Stage 05",
                "Thematic Visualization and Grid-Based Prescription Mapping",
                "Compact prescription-mapping layout.",
            )
        )

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Group(elem_classes="frame-card"):
                    gr.HTML(
                        render_box_header(
                            "GeoTIFF Visualization",
                            "Render a publication-ready thematic map",
                            "",
                        )
                    )
                    tif_file = gr.File(
                        label="Upload a GeoTIFF file",
                        file_types=[".tif", ".tiff"],
                    )
                    colormap = gr.Dropdown(
                        choices=COLORMAP_OPTIONS,
                        value="RdYlGn",
                        label="Colormap",
                    )
                    clip_slider = gr.Slider(
                        minimum=0,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Percentile clipping (%)",
                    )
                    vis_btn = gr.Button("Generate the thematic visualization", variant="primary")
                    output_img = gr.Image(
                        label="Visualization result",
                        type="pil",
                        format="png",
                        elem_classes="portrait-image",
                    )
            with gr.Column(scale=7):
                with gr.Group(elem_classes=["frame-card", "frame-output"]):
                    gr.HTML(
                        render_box_header(
                            "Grid Prescription Map",
                            "Create grid-level management zones",
                            "",
                        )
                    )
                    tif_file_input = gr.File(
                        label="Upload a nitrogen prediction TIF file",
                        file_types=[".tif", ".tiff"],
                    )
                    with gr.Row():
                        grid_width = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Grid width (m)",
                        )
                        N_std = gr.Number(label="Standard nitrogen concentration (g/kg)")
                        B_std = gr.Number(label="Standard biomass (kg/ha)")
                    grid_btn = gr.Button("Generate the grid prescription map", variant="primary")
                    output_plot = gr.Plot(label="Grid prescription plot")
                    grid_output_path = gr.Textbox(
                        label="Grid output path",
                        interactive=False,
                        elem_classes="path-box",
                    )

    return {
        "tif_file": tif_file,
        "colormap": colormap,
        "clip_slider": clip_slider,
        "vis_btn": vis_btn,
        "output_img": output_img,
        "tif_file_input": tif_file_input,
        "grid_width": grid_width,
        "N_std": N_std,
        "B_std": B_std,
        "grid_btn": grid_btn,
        "output_plot": output_plot,
        "grid_output_path": grid_output_path,
    }


with gr.Blocks(css=APP_CSS, title="AgriMapper V13") as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(render_hero())

        with gr.Tabs():
            data_tab = create_data_input_tab()
            feature_tab = create_feature_engineering_tab()
            model_tab = create_model_training_tab()
            estimation_tab = create_estimation_tab()
            prescription_tab = create_prescription_tab()

    data_tab["upload_btn"].click(
        fn=generate_components,
        inputs=[data_tab["tif_input"], data_tab["file_input"]],
        outputs=[feature_tab["window_size_input"]] + feature_tab["dropdowns"] + [feature_tab["dataset_checkbox"]],
    )

    feature_tab["submit_btn"].click(
        fn=STEP1_data_process_main,
        inputs=[
            data_tab["tif_input"],
            data_tab["file_input"],
            data_tab["output_dir"],
            data_tab["target_value"],
            feature_tab["window_size_input"],
            *feature_tab["dropdowns"],
            feature_tab["dataset_checkbox"],
        ],
        outputs=[
            feature_tab["features_df"],
            feature_tab["sorted_features"],
            feature_tab["gallery"],
            feature_tab["band_order"],
        ],
    )

    model_tab["model_option"].change(
        fn=model_data_upload,
        inputs=model_tab["model_option"],
        outputs=model_tab["file_upload"],
    )

    model_tab["submit_btn"].click(
        fn=STEP2_model_train_main,
        inputs=[
            model_tab["model_option"],
            model_tab["file_upload"],
            feature_tab["features_df"],
            data_tab["output_dir"],
            model_tab["validation_mode"],
            model_tab["cv_model_group"],
        ],
        outputs=[
            model_tab["performance_df"],
            model_tab["gallery"],
            model_tab["model_path"],
        ],
    )

    estimation_tab["submit_btn"].click(
        fn=STEP3_Estimation_map_main,
        inputs=[
            estimation_tab["multi_tif_path"],
            model_tab["model_path"],
            estimation_tab["model_option"],
            feature_tab["band_order"],
            feature_tab["sorted_features"],
            data_tab["output_dir"],
        ],
        outputs=[estimation_tab["gallery"], estimation_tab["output_txt"]],
    )

    estimation_tab["index_btn"].click(
        fn=STEP4_indices_map_main,
        inputs=[
            estimation_tab["index_tif_path"],
            feature_tab["band_order"],
            estimation_tab["index_dropdown"],
            data_tab["output_dir"],
        ],
        outputs=[estimation_tab["output_index_txt"], estimation_tab["histogram"]],
    )

    prescription_tab["vis_btn"].click(
        fn=STEP5_visualize_tif,
        inputs=[
            prescription_tab["tif_file"],
            prescription_tab["colormap"],
            prescription_tab["clip_slider"],
        ],
        outputs=prescription_tab["output_img"],
    )

    prescription_tab["grid_btn"].click(
        fn=STEP6_grid_main,
        inputs=[
            prescription_tab["tif_file_input"],
            prescription_tab["grid_width"],
            prescription_tab["N_std"],
            prescription_tab["B_std"],
            data_tab["output_dir"],
        ],
        outputs=[prescription_tab["output_plot"], prescription_tab["grid_output_path"]],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()
