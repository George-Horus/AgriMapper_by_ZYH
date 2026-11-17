# AgriMapper

An interactive Python and Gradio platform for crop phenotyping, nitrogen estimation, and prescription-map generation from aerial multispectral imagery.

Experimental data used during software development is available in the [GitHub Releases](https://github.com/George-Horus/AgriMapper_by_ZYH/releases/tag/1.0).

A related research article has been accepted for publication: [https://doi.org/10.1016/j.ecoinf.2026.104128](https://doi.org/10.1016/j.ecoinf.2026.104128).

## Features

- **End-to-end workflow**: upload imagery and field observations, extract spectral features, train models, generate prediction maps, and create grid-based prescription maps.
- **Vegetation-index library**: calculate 117 multispectral vegetation indices when the required bands are available.
- **Model comparison**: train 15 models across traditional machine learning, deep learning, and AutoML:
  - 9 traditional ML models: MLR, PLSR, KNN, SVR, Random Forest, GBDT, XGBoost, ELM, and BPNN.
  - 4 deep-learning models: DNN, MLP, CNN, and Autoencoder.
  - 2 AutoML models: AutoGluon and TPOT.
- **Validation options**: run the standard train/test workflow or five-fold cross-validation. Cross-validation writes fold-level evaluation artifacts; standard training is required to create the model files used for field prediction.
- **Large-image prediction**: process GeoTIFF imagery tile by tile to limit memory use.
- **Map outputs**: export prediction GeoTIFFs, histograms, thematic visualizations, and grid-based prescription maps.

## Workflow

The application is organized into five tabs:

1. **01 Data Input** - upload multispectral GeoTIFF files and field sampling files, set the output directory, and specify the target column.
2. **02 Feature Engineering** - map bands, set the sampling window, select datasets, calculate indices, and perform Pearson filtering plus RF-RFE feature selection.
3. **03 Model Training** - train the 15 models, compare performance, or run five-fold cross-validation.
4. **04 Estimation Results** - use a trained ML, DL, or ensemble model to generate pixel-wise prediction maps and index maps.
5. **05 Prescription Mapping** - visualize a prediction GeoTIFF and generate a grid-level prescription map.

The feature filtering stage is completed before the five-fold modelling branch. For an unbiased estimate that includes feature-selection uncertainty, repeat feature selection inside each training fold or use an independent external test set.

![AgriMapper workflow](./images/results.PNG)

*Figure 1. Representative AgriMapper workflow.*

![AgriMapper interface](./images/interface.png)

*Figure 2. AgriMapper interface.*

> The images in `images/` are representative workflow screenshots. The current V13 application uses the five tab names listed above; some labels in the screenshots reflect the earlier interface layout.

## Installation

The current project is intended to run locally with Python. Python 3.12 is recommended. Dependencies are listed in [`requirements.txt`](./requirements.txt).

```bash
# Clone the repository
git clone https://github.com/George-Horus/AgriMapper_by_ZYH.git
cd AgriMapper_by_ZYH/Agrimapper2.0

# Create and activate an environment
conda create -n agrimapper python=3.12
conda activate agrimapper

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Launch the current interface
python Gradio_V13.py
```

Open the local URL printed by Gradio in a browser. The current repository does not include a Dockerfile or a maintained Docker image, so the old Docker Hub commands are not part of the installation instructions.

## Input data

### Multispectral imagery

- Use one or more GeoTIFF files (`.tif` or `.tiff`).
- In **02 Feature Engineering**, map each raster band to the correct name and preserve the same order for prediction. Supported band names are `BLUE`, `GREEN`, `NIR`, `RED`, `RedEdge`, and `other`.
- Images should use a valid coordinate reference system. Point coordinates are interpreted as WGS84 longitude and latitude and transformed to the image CRS.

### Field sampling data

Supported formats are `.csv`, `.xlsx`, `.txt`, and point `.shp` files.

- CSV and Excel files must contain `lon` and `lat` columns and the measured target column.
- TXT files are read as whitespace-separated coordinate columns. Because target-column parsing is more restrictive for TXT input, CSV, Excel, or a point shapefile is recommended when the same file must carry both coordinates and measured values.
- Point shapefiles must contain point geometry and the target column.
- Enter the exact target-column name in **01 Data Input**.

The sampling window controls whether a single pixel or a local window mean is extracted around each point. Confirm that the sampling points overlap the uploaded imagery before starting feature engineering.

When each image has its own sampling table, upload matching numbers of images and tables. When several dates share one sampling table, upload one table together with the image set.

## Usage

### 1. Prepare the data

Prepare co-registered multispectral GeoTIFF files and the corresponding field sampling table. Crop or mask the imagery to the study area when appropriate.

![Input files](./images/Step2.PNG)

*Figure 3. Representative file-upload screen.*

### 2. Run feature engineering

Upload the files in **01 Data Input**, click **Read files and generate the parameter panel**, map the bands in **02 Feature Engineering**, choose the datasets to include, set the sampling window, and click **Run feature filtering and export results**.

The stage writes the extracted table, selected-feature table, and feature-selection figures to:

```text
<output directory>/Feature_Selection/
```

![Data input and feature engineering](./images/Step1.PNG)

*Figure 4. Representative input preparation and feature-engineering workflow.*

![Feature filtering](./images/Step3.PNG)

*Figure 5. Representative feature-filtering screen.*

### 3. Train and compare models

In **03 Model Training**, either use the fixed-feature table produced by Stage 02 or upload a custom `.csv`/`.xlsx` modelling table. The first column must be the target variable and the remaining columns must be numeric predictors.

- **Standard train/test comparison** writes deployable best ML, DL, and AutoML model files under `Best_Model/`.
- **Five-fold cross-validation** writes `fold_metrics.csv`, `oof_predictions.csv`, `model_summary.csv`, model-status records, figures, and an Excel workbook under `Cross_Validation/`. These evaluation artifacts are not deployable prediction models.

![Model training](./images/Step4.PNG)

*Figure 6. Representative model-training interface and outputs.*

### 4. Generate estimation maps

After standard training, open **04 Estimation Results**, select the best ML, DL, or ensemble model, upload one or more target-field GeoTIFF files, and click **Generate prediction maps and summary figures**. Prediction GeoTIFFs, histograms, and logs are saved under:

```text
<output directory>/Prediction_Results/
```

The same tab can calculate and export a single vegetation-index map.

![Estimation results](./images/Step5.PNG)

*Figure 7. Representative estimation-result interface.*

### 5. Generate a prescription map

In **05 Prescription Mapping**:

1. Upload a prediction GeoTIFF and choose a colormap.
2. Set the percentile-clipping value and generate the thematic visualization.
3. Upload the nitrogen prediction GeoTIFF for prescription mapping.
4. Set the grid width, standard nitrogen concentration, and standard biomass.
5. Click **Generate the grid prescription map**.

The AGN GeoTIFF and PNG grid map are saved under:

```text
<output directory>/grid_map/
```

![Prescription mapping](./images/Step6.PNG)

*Figure 8. Representative prescription-mapping interface.*

The standard nitrogen concentration, standard biomass, fertilizer parameters, and decision thresholds should be calibrated for the target crop and field before operational use.

## Repository layout

```text
Agrimapper2.0/
├── Gradio_V13.py                 # Gradio entry point
├── requirements.txt              # Python dependencies
├── S1_preprocessing/             # Band extraction and feature engineering
├── S2_model_training/            # ML, DL, AutoML, and cross-validation
├── S3_prediction_vis/            # Prediction, visualization, and prescription maps
├── script/                       # Optional data and raster utilities
└── images/                       # README workflow screenshots
```

## Notes

- Prediction requires models produced by standard training. A cross-validation run alone does not populate the prediction tab with deployable model paths.
- Feature selection currently occurs before the standard five-fold modelling call. Treat the resulting scores as internal validation results unless feature selection is nested within the folds or an external test set is used.
- Model performance depends on sensor bands, image preprocessing, sampling design, crop, field conditions, and target variable. Validate model outputs and prescription parameters locally before field application.
