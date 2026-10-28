# Frozen label mapping and dataset audit

Pinned revision: `5467f6012d78d1c446145d5f582da6096f852ae8`. Counts are Git tree image-file counts, not decoded/eligible observations. Both supplied folders are external-only. Actual eligibility additionally requires the mapped PlantVillage class, readable images, and passage through duplicate/overlap exclusions. Do not change this mapping based on model test performance.

## PlantVillage color-image class distribution

Pinned revision: `9e97599868962bd0079b8db4b7f1efa9185fa1e7`. All 54,305 listed color images decoded successfully; these are the audited per-class totals used by the source split preparation.

| PlantVillage class | Images |
|---|---:|
| `Apple___Apple_scab` | 630 |
| `Apple___Black_rot` | 621 |
| `Apple___Cedar_apple_rust` | 275 |
| `Apple___healthy` | 1,645 |
| `Blueberry___healthy` | 1,502 |
| `Cherry_(including_sour)___Powdery_mildew` | 1,052 |
| `Cherry_(including_sour)___healthy` | 854 |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 513 |
| `Corn_(maize)___Common_rust_` | 1,192 |
| `Corn_(maize)___Northern_Leaf_Blight` | 985 |
| `Corn_(maize)___healthy` | 1,162 |
| `Grape___Black_rot` | 1,180 |
| `Grape___Esca_(Black_Measles)` | 1,383 |
| `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | 1,076 |
| `Grape___healthy` | 423 |
| `Orange___Haunglongbing_(Citrus_greening)` | 5,507 |
| `Peach___Bacterial_spot` | 2,297 |
| `Peach___healthy` | 360 |
| `Pepper,_bell___Bacterial_spot` | 997 |
| `Pepper,_bell___healthy` | 1,478 |
| `Potato___Early_blight` | 1,000 |
| `Potato___Late_blight` | 1,000 |
| `Potato___healthy` | 152 |
| `Raspberry___healthy` | 371 |
| `Soybean___healthy` | 5,090 |
| `Squash___Powdery_mildew` | 1,835 |
| `Strawberry___Leaf_scorch` | 1,109 |
| `Strawberry___healthy` | 456 |
| `Tomato___Bacterial_spot` | 2,127 |
| `Tomato___Early_blight` | 1,000 |
| `Tomato___Late_blight` | 1,909 |
| `Tomato___Leaf_Mold` | 952 |
| `Tomato___Septoria_leaf_spot` | 1,771 |
| `Tomato___Spider_mites Two-spotted_spider_mite` | 1,676 |
| `Tomato___Target_Spot` | 1,404 |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5,357 |
| `Tomato___Tomato_mosaic_virus` | 373 |
| `Tomato___healthy` | 1,591 |

| PlantDoc folder | Train-folder files (external only) | Test-folder files | PlantVillage class or exclusion |
|---|---:|---:|---|
| Apple Scab Leaf | 83 | 10 | `Apple___Apple_scab` |
| Apple leaf | 82 | 9 | `Apple___healthy` |
| Apple rust leaf | 79 | 10 | `Apple___Cedar_apple_rust` |
| Bell_pepper leaf | 53 | 8 | `Pepper,_bell___healthy` |
| Bell_pepper leaf spot | 62 | 9 | Excluded: Symptom label insufficient to establish pathogen equivalence |
| Blueberry leaf | 106 | 11 | `Blueberry___healthy` |
| Cherry leaf | 47 | 10 | `Cherry_(including_sour)___healthy` |
| Corn Gray leaf spot | 64 | 4 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` |
| Corn leaf blight | 180 | 12 | Excluded: Blight subtype not specified |
| Corn rust leaf | 106 | 10 | `Corn_(maize)___Common_rust_` |
| Peach leaf | 103 | 9 | `Peach___healthy` |
| Potato leaf early blight | 109 | 8 | `Potato___Early_blight` |
| Potato leaf late blight | 97 | 8 | `Potato___Late_blight` |
| Raspberry leaf | 112 | 7 | `Raspberry___healthy` |
| Soyabean leaf | 57 | 8 | `Soybean___healthy` |
| Squash Powdery mildew leaf | 124 | 6 | `Squash___Powdery_mildew` |
| Strawberry leaf | 88 | 8 | `Strawberry___healthy` |
| Tomato Early blight leaf | 79 | 9 | `Tomato___Early_blight` |
| Tomato Septoria leaf spot | 140 | 11 | `Tomato___Septoria_leaf_spot` |
| Tomato leaf | 55 | 8 | `Tomato___healthy` |
| Tomato leaf bacterial spot | 101 | 9 | `Tomato___Bacterial_spot` |
| Tomato leaf late blight | 101 | 10 | `Tomato___Late_blight` |
| Tomato leaf mosaic virus | 44 | 10 | `Tomato___Tomato_mosaic_virus` |
| Tomato leaf yellow virus | 70 | 6 | Excluded: Symptom label insufficient to establish pathogen equivalence |
| Tomato mold leaf | 85 | 6 | Excluded: Mold etiology not specified |
| Tomato two spotted spider mites leaf | 2 | 0 | Excluded: Pest identity/cropped-label spelling needs manual review |
| grape leaf | 57 | 12 | `Grape___healthy` |
| grape leaf black rot | 56 | 8 | `Grape___Black_rot` |

Total: 2,578 files across 28 labels. All 2,578 images decoded successfully; the audit found no corrupt files, 17 pixel-duplicate groups and 47 dHash candidate groups. The frozen mapping yielded 2,146 pre-screening candidates in 23 labels. After duplicate and source-overlap screening, 2,107 images remain eligible; 39 duplicate external images were excluded and no source/external overlap was found. The remaining 432 of 471 total exclusions were labels not mapped because their folder descriptions do not establish a safe class match. Healthy labels follow the dataset annotation; health has not been independently verified. Semantic correspondence is not expert adjudication.

## Eligible external images after screening

These are the observed PlantDoc image counts that enter each manifest's external evaluation split, after applying the frozen mapping and duplicate/overlap checks. All are held out from training, validation-based calibration and threshold selection.

| PlantVillage class | Eligible PlantDoc images |
|---|---:|
| `Apple___Apple_scab` | 92 |
| `Apple___Cedar_apple_rust` | 89 |
| `Apple___healthy` | 91 |
| `Blueberry___healthy` | 116 |
| `Cherry_(including_sour)___healthy` | 57 |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 68 |
| `Corn_(maize)___Common_rust_` | 116 |
| `Grape___Black_rot` | 63 |
| `Grape___healthy` | 69 |
| `Peach___healthy` | 112 |
| `Pepper,_bell___healthy` | 61 |
| `Potato___Early_blight` | 114 |
| `Potato___Late_blight` | 102 |
| `Raspberry___healthy` | 118 |
| `Soybean___healthy` | 65 |
| `Squash___Powdery_mildew` | 129 |
| `Strawberry___healthy` | 96 |
| `Tomato___Bacterial_spot` | 108 |
| `Tomato___Early_blight` | 81 |
| `Tomato___Late_blight` | 100 |
| `Tomato___Septoria_leaf_spot` | 143 |
| `Tomato___Tomato_mosaic_virus` | 54 |
| `Tomato___healthy` | 63 |

The 432 unmappable-label exclusions are: Bell pepper leaf spot (71) plus tomato leaf yellow virus (76), both excluded because the symptom/label does not establish pathogen equivalence (147 combined); corn leaf blight with unspecified subtype (192); tomato mold with unspecified etiology (91); and the spider-mite folder requiring manual review (2). The other 39 exclusions are duplicate external images.

## Acquisition and manifest audit

Pinned PlantDoc revision `5467f6012d78d1c446145d5f582da6096f852ae8` contains 2,578 image files in 28 folder labels. Windows-invalid names and case-insensitive collisions were preserved as distinct files using 93 recorded path mappings; the resulting extracted inventory matches the pinned per-folder counts. PlantVillage revision `9e97599868962bd0079b8db4b7f1efa9185fa1e7` contains 54,305 color images in 38 classes, with no corrupt files. The experiment's three manifests have identical fixed split membership: 37,863 train, 8,256 validation, 8,186 in-domain test and 2,107 external images. Their path, SHA-256, pixel, dHash and source-leaf overlap checks passed.

The PlantVillage leaf map has 40,328 official keys matching 41,111 image rows. The remaining 13,194 rows across 15 classes use the explicitly configured class-scoped filename fallback. A unique upstream mapping is retained for 621 Apple Black Rot filenames whose map entry uses the legacy `Apple_Frogeye Spot` alias. This is grouping metadata only; disease labels are unchanged. Filename-derived IDs can group repeated suffixes but cannot guarantee related physical leaves are always separated. This limitation must accompany any final interpretation.

Source: [pinned Cropped PlantDoc tree](https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/git/trees/5467f6012d78d1c446145d5f582da6096f852ae8?recursive=1). The audited per-class PlantVillage counts and PlantDoc folders are saved under `D:/plant-disease-research-data/full/artifacts/reports/full_seed42/`; the pinned archive and source metadata records are under `D:/plant-disease-research-data/full/data/raw/`.
