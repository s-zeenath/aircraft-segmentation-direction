# Aircraft Vision

Aircraft detection, instance segmentation, and heading estimation for aerial imagery.

**[Open the live app](https://aircraft-segmentation-direction.streamlit.app)**

Upload an aerial image or use one of the included examples. Toggle detection boxes, segmentation masks, nose/tail keypoints, and heading arrows; adjust the detection confidence threshold interactively.

## Validation results

The models were trained for 20 epochs at 640 px on 500 images (400 training and 100 validation), with 5,288 aircraft annotations.

| Task | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Aircraft detection | 0.967 | 0.896 | 0.960 | 0.755 |
| Instance segmentation | 0.928 | 0.842 | 0.920 | 0.525 |
| Nose/tail keypoints | 0.952 | 0.863 | 0.937 | 0.884 |

On the validation split, heading evaluation matched 852 of 971 aircraft using a bounding-box IoU threshold of 0.50. Mean angular error was 13.75°, median error was 5.44°, and 85.21% of matched headings were within 20°.

A previously recorded inference benchmark was 38.8 ms per image. Hardware details were not recorded, so actual inference time varies by host.

## Run locally

Requires Python 3.11. Install the dependencies and launch the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app loads `models/segmentation.pt` and `models/pose.pt`. Keep both model files and the `app_assets/examples/` directory in the repository.

## Deployment

The app is hosted on [Streamlit Community Cloud](https://share.streamlit.io/). To deploy your own instance, push the project to GitHub, select the repository, and set `app.py` as the entry point. Dependencies are listed in `requirements.txt`.

Training datasets, archives, and experiment outputs are not included because they are not required to run the app.
