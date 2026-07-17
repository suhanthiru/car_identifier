# Datasets

All real-data evaluation runs on established public research datasets. Every
loader takes a root path from `datasets/config.py` (env-overridable) and hard-
gates on the data actually being present — **no dataset, no numbers**; the eval
harness writes "PENDING" sections instead of fabricating anything.

I cannot agree to research-use terms on your behalf, so each dataset below
needs a manual request/download by you. Place them under `data/datasets/` (or
set the env var) with the exact layouts shown.

## VeRi-776 (start here — plates + color/type attributes)

- ~50k images, 776 vehicle identities, 20 cameras, with color/type labels and
  license-plate annotations. The standard vehicle-ReID benchmark.
- **How to get it**: request via the authors' form (search "VeRi dataset
  request" — maintained by Xinchen Liu et al.; a Google Form linked from
  `github.com/JDAI-CV/VeRidataset`). You agree to research-only terms; they
  send a download link.
- Expected layout (`EYES_VERI_ROOT`, default `data/datasets/VeRi`):

```
VeRi/
├── image_train/          0002_c002_00030600_0.jpg ...
├── image_query/
├── image_test/
├── name_train.txt
├── name_query.txt
├── name_test.txt
├── train_label.xml       color/type per vehicle id
├── test_label.xml
└── (optional) camera_ID.txt, gt_index.txt, jk_index.txt
```

## VehicleID (second retrieval benchmark)

- 220k+ images, 26k identities, front/rear views only.
- **How to get it**: request from the PKU authors ("PKU VehicleID dataset
  request" — via `pkuml.org/resources/pku-vehicleid.html`).
- Layout (`EYES_VEHICLEID_ROOT`, default `data/datasets/VehicleID`):

```
VehicleID/
├── image/                0000001.jpg ...
├── train_test_split/
│   ├── train_list.txt    "<image> <vehicle_id>" per line
│   ├── test_list_800.txt
│   ├── test_list_1600.txt
│   └── test_list_2400.txt
└── attribute/            (color/model if included in your release)
```

## CityFlow / AI City Challenge (multi-camera geometry + trajectories)

- Real multi-camera traffic video with camera calibrations, timestamps, and
  cross-camera vehicle trajectories — what the transit-time veto and
  corroboration validation need.
- **How to get it**: as of mid-2026 the AI City Challenge removed the
  request-form/password gate. The relevant package is **2022 Track 1:
  City-Scale Multi-Camera Vehicle Tracking** (CityFlowV2), linked from
  `aicitychallenge.org/ai-city-challenge-dataset-access/` → the
  2022-track1-download page (Google Drive). Downloading still means
  accepting their *Dataset License AIC2022* (PDF on that page) — read it;
  it restricts use to non-commercial research. The 2021 Track 2 ReID
  package on the same page is a useful optional extra.
- Layout (`EYES_CITYFLOW_ROOT`, default `data/datasets/CityFlow`):

```
CityFlow/
├── train/
│   └── S01/
│       ├── c001/
│       │   ├── vdo.avi
│       │   ├── calibration.txt    homography  + (optional) GPS
│       │   └── gt/gt.txt          MOT format: frame,id,x,y,w,h,...
│       └── c002/ ...
└── validation/ ...
```

## Optional: VeRi-Wild

Same request-form pattern (`github.com/PKU-IMRE/VERI-Wild`). Loader not built
yet; added on demand.

## Not acceptable

Scraped surveillance feeds, covert footage of identifiable people, any
non-consented real-world camera data. The loaders will not be extended to
ingest such sources.
