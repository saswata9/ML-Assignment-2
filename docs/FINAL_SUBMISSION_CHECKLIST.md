# Final publication and submission checklist

## 1. Project validation
- [x] Model-training script creates five trained model artifacts in `model/` on first app launch.
- [x] Model-training script generates the 114-row held-out `test_data.csv` on first app launch.
- [x] Metrics saved in `model_comparison.csv`.
- [x] Streamlit app source included in `app.py`.
- [x] Dependencies listed in `requirements.txt`.
- [x] Saved model predictions rechecked against reported metrics.
- [x] Python syntax checked for `app.py` and `train_models.py`.

## 2. GitHub publication
- Repository: https://github.com/saswata9/ML-Assignment-2
- Default branch: `main`
- Confirm that all project files and the five files under `model/` are present.

## 3. Deploy on Streamlit Community Cloud
1. Sign in to Streamlit Community Cloud using the GitHub account containing this repository.
2. Choose **Create app** and select repository `saswata9/ML-Assignment-2`.
3. Select branch `main` and main file path `app.py`.
4. Deploy and wait for the build to finish.
5. Open the public app URL and test it in a private/incognito browser window.
6. Upload the included `test_data.csv`; verify all five models produce metrics, then inspect each model's confusion matrix and classification report.
7. Test the predictions download button.
8. Add the real public app URL to the README after deployment.

## 4. Capture the required BITS Virtual Lab screenshot
1. Sign in to the official BITS Virtual Lab environment required by the course.
2. Open/run the project there, following the course's allowed workflow.
3. Launch the app and upload `test_data.csv`.
4. Capture a genuine screenshot showing the running application and relevant model metrics.
5. Save the screenshot in `docs/` and include it in the final submission if required.

## 5. Final submission checks
- [x] Repository link is published: https://github.com/saswata9/ML-Assignment-2
- [ ] Deployed app link opens and works.
- [ ] App has been tested using `test_data.csv` on Streamlit Cloud.
- [ ] Genuine BITS Virtual Lab screenshot is included.
- [ ] README/report contains the actual app URL once deployed.
- [ ] You can explain the dataset, target encoding, train/test split, all six metrics, and model comparison.
