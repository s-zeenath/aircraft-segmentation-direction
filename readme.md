# Aircraft Vision

Aircraft detection, instance segmentation, and heading estimation in an interactive Streamlit app. Upload an aerial image or try one of the included examples.

## Validation results

The models were trained for 20 epochs at 640 px on 500 images (400 training, 100 validation; 5,288 aircraft annotations).

| Task | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Detection | 0.967 | 0.896 | 0.960 | 0.755 |
| Instance segmentation | 0.928 | 0.842 | 0.920 | 0.525 |
| Nose/tail keypoints | 0.952 | 0.863 | 0.937 | 0.884 |

Heading evaluation matched 852 of 971 validation aircraft at bounding-box IoU >= 0.50. Mean angular error was 13.75 degrees and median error was 5.44 degrees; 85.21% of matched headings were within 20 degrees. A previously recorded inference benchmark was 38.8 ms/image; hardware details were not recorded, so runtime will vary by host.

## Run locally

Use Python 3.11, install dependencies, then start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app loads `models/segmentation.pt` and `models/pose.pt`. Keep both files and the `app_assets/examples/` directory in the repository.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and create an app from the repository, selecting `app.py` as the entry point.
3. Deploy. Dependencies are listed in `requirements.txt`; model weights and example images are included in the repository.

The training datasets, archives, and experiment outputs are intentionally excluded; they are not needed to run the app.
