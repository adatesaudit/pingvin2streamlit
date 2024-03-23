import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Penguin finder", page_icon=":penguin:") # , layout="wide")

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
        "Pingvin lelőhelye (sziget) - island",
        options=["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox(
        "Neme - sex", options=["Female", "Male"])
    bill_length = st.number_input(
        "Csőr hossz (mm) - bill lenght", min_value=0)
    bill_depth = st.number_input(
        "Csőr átmérő (mm) - bill depth", min_value=0)
    flipper_length = st.number_input(
        "Uszony hossz (mm) - flipper lenght", min_value=0)
    body_mass = st.number_input(
        "Testtömeg (g) - body mass", min_value=0)
    st.form_submit_button("Okézd le")
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
    """Gépi tanuló modell használtunk 
    (Random Forest) hogy előre tudjuk jelezni 
    a fajtákat, az alábbi képen a beállítható jellemzők
    relatív fontossági sorrendben vannak bemutatva."""
)
st.image("feature_importance.png")

# todo another new part of script
st.write(
    """Az alábbiakban az egyes folytonos változók hisztogramjai láthatóak pingvinfajok szerint elkülönítve. 
    A függőleges vonal a beírt értéket jelöli."""
)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["bill_length_mm"],
    hue=penguin_df["species"])
plt.axvline(bill_length)
plt.title("Csőr hosszúság fajtánként - Bill Length by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["bill_depth_mm"],
    hue=penguin_df["species"])
plt.axvline(bill_depth)
plt.title("Csőr átmérő fajtánként - Bill Depth by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(
    x=penguin_df["flipper_length_mm"],
    hue=penguin_df["species"])
plt.axvline(flipper_length)
plt.title("Uszony hosszúság fajtánként - Flipper Length by Species")
st.pyplot(ax)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)



