import streamlit as st
import pandas as pd
import joblib

# Page title
st.title("House Price Prediction")

# Description
st.write(
    "Enter the house details below to estimate its selling price."
)

# Load model and preprocessor
model = joblib.load("house_price_app_model.pkl")
preprocessor = joblib.load("app_preprocessor.pkl")
st.subheader("Enter House Details")

# Create two columns
col1, col2 = st.columns(2)

# ---------------- COLUMN 1 ----------------
with col1:

    gr_liv_area = st.number_input(
        "Living Area (GrLivArea)",
        min_value=0,
        key="gr_liv_area"
    )

    overall_qual = st.number_input(
        "Overall Quality (1-10)",
        min_value=1,
        max_value=10,
        value=5,
        key="overall_qual"
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=0,
        max_value=10,
        value=3,
        key="bedrooms"
    )

    full_bath = st.number_input(
        "Number of Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        key="full_bath"
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000,
        key="year_built"
    )

    garage_cars = st.number_input(
        "Garage Capacity (Cars)",
        min_value=0,
        max_value=5,
        value=2,
        key="garage_cars"
    )

    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        value=500,
        key="garage_area"
    )


# ---------------- COLUMN 2 ----------------
with col2:

    total_bsmt_sf = st.number_input(
        "Total Basement Area (sq ft)",
        min_value=0,
        value=1000,
        key="total_bsmt_sf"
    )

    first_floor_sf = st.number_input(
        "1st Floor Area (sq ft)",
        min_value=0,
        value=1000,
        key="first_floor_sf"
    )

    second_floor_sf = st.number_input(
        "2nd Floor Area (sq ft)",
        min_value=0,
        value=500,
        key="second_floor_sf"
    )

    total_rooms = st.number_input(
        "Total Rooms Above Ground",
        min_value=0,
        max_value=20,
        value=6,
        key="total_rooms"
    )

    year_remod = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2000,
        key="year_remod"
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        [
            "CollgCr",
            "Veenker",
            "Crawfor",
            "NoRidge",
            "Mitchel",
            "Somerst",
            "NWAmes",
            "OldTown",
            "BrkSide",
            "Sawyer",
            "NridgHt",
            "NAmes",
            "SawyerW",
            "IDOTRR",
            "MeadowV",
            "Edwards",
            "Timber",
            "Gilbert",
            "StoneBr",
            "ClearCr",
            "NPkVill",
            "Blmngtn",
            "BrDale",
            "SWISU",
            "Blueste"
        ],
        key="neighborhood"
    )

    year_sold = st.number_input(
        "Year Sold",
        min_value=2006,
        max_value=2010,
        value=2008,
        key="year_sold"
    )


# Prediction button
if st.button("Predict House Price"):

    input_data = pd.DataFrame([{
        "GrLivArea": gr_liv_area,
        "OverallQual": overall_qual,
        "BedroomAbvGr": bedrooms,
        "FullBath": full_bath,
        "YearBuilt": year_built,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "TotalBsmtSF": total_bsmt_sf,
        "1stFlrSF": first_floor_sf,
        "2ndFlrSF": second_floor_sf,
        "TotRmsAbvGrd": total_rooms,
        "YearRemodAdd": year_remod,
        "Neighborhood": neighborhood,
        "YrSold": year_sold
    }])

    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)

    st.success(
        f"Estimated House Price: ${prediction[0]:,.2f}"
    )