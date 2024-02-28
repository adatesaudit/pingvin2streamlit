import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
st.title('Pingvin kategorizáló: Egy gépi tanuló app')
st.write("Ez az alkalmazás 6 bemeneti adatot használ a pingvinfaj előrejelzéséhez a "
         "a Palmer Pingvinek adathalmazán alapuló modell alapján. Használja az űrlapot"
         " hogy elkezdhesse!")
penguin_df = pd.read_csv('penguins.csv')
rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()
##############
with st.sidebar.form("user_inputs"):
    island = st.selectbox(
        "Pingvin lelőhelye (sziget)",
        options=["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox(
        "Neme", options=["Female", "Male"])
    bill_length = st.number_input(
        "Csőr hossz (mm)", min_value=0)
    bill_depth = st.number_input(
        "Csőr átmérő (mm)", min_value=0)
    flipper_length = st.number_input(
        "Uszony hossz (mm)", min_value=0)
    body_mass = st.number_input(
        "Testtömeg (g)", min_value=0)
    st.form_submit_button()
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1
sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1
new_prediction = rfc.predict(
    [
        [
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            island_biscoe,
            island_dream,
            island_torgerson,
            sex_female,
            sex_male,
        ]
    ]
)
prediction_species = unique_penguin_mapping[new_prediction][0]
# st.write(f"Azt gondoljuk, hogy ez egy {prediction_species} fajta adata.")

# todo additional part of article
st.subheader("Pingvin fajta megjósolása:")
st.write(f"Azt gondoljuk, hogy ennek {prediction_species} fajtának kell lennie")
st.write(
    """We used a machine learning 
    (Random Forest) model to predict the 
    species, the features used in this 
    prediction are ranked by relative 
    importance below."""
)
st.image("feature_importance.png")

# todo another new part of script
st.write(
    """Below are the histograms for each
continuous variable separated by penguin species.
The vertical line represents the inputted value."""
)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["bill_length_mm"],
    hue=penguin_df["species"])
plt.axvline(bill_length)
plt.title("Bill Length by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["bill_depth_mm"],
    hue=penguin_df["species"])
plt.axvline(bill_depth)
plt.title("Bill Depth by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["flipper_length_mm"],
    hue=penguin_df["species"])
plt.axvline(flipper_length)
plt.title("Flipper Length by Species")
st.pyplot(ax)
