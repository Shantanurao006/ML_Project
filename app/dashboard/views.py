from pathlib import Path
import pandas as pd
import joblib
from django.shortcuts import render

BASE_DIR = Path(__file__).resolve().parent.parent.parent

model = joblib.load(BASE_DIR / "outputs/models/final_model.pkl")

def home(request):
    comparison_df = pd.read_csv(
        BASE_DIR / "outputs/comparison/final_comparison.csv"
    )

    comparison_df = comparison_df.rename(columns={
        "F1 Score": "F1_Score",
        "Training Time": "Training_Time"
    })

    pyspark_df = pd.read_csv(
    BASE_DIR / "outputs/comparison/pyspark_vs_pandas.csv"
    )

    pyspark_df = pyspark_df.rename(columns={
        "Rows Processed": "Rows_Processed",
        "Time Seconds": "Time_Seconds"
    })

    context = {
        "results": comparison_df.to_dict(orient="records"),
        "pyspark_results": pyspark_df.to_dict(orient="records"),
        "best_model": "Approach 3 - Random Forest Without Leakage"
    }

    return render(request, "dashboard/home.html", context)


def predict(request):
    prediction = None

    if request.method == "POST":
        brand = int(request.POST.get("brand"))
        category = int(request.POST.get("category"))
        price = float(request.POST.get("price"))
        hour = int(request.POST.get("hour"))

        input_data = [[brand, category, price, 10, 1, hour, 1]]

        pred = model.predict(input_data)[0]

        prediction = (
            "Purchase Likely"
            if pred == 1
            else "Purchase Not Likely"
        )

    return render(
        request,
        "dashboard/predict.html",
        {"prediction": prediction}
    )