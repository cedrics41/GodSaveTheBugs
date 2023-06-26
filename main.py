#from html.parser import HTMLParser
#from turtle import color
import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px
import datetime
from datetime import date
import calendar
import time
from PIL import Image#, ImageGrab
from zipfile import ZipFile
from base64 import b64encode
from fpdf import FPDF, HTMLMixin
#import tempfile
import plotly.graph_objects as go
import plotly.subplots as sp
import base64
import os
from tabulate import tabulate



# from Master_Project_report_page import report

st.set_page_config(layout="wide")


# -------------------------------------------------------------------------------
# Background
def add_background(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


# -------------------------------------------------------------------------------
# Read an uploaded file (csv format)
# st.cache_data
def read_csv(file):
    df = pd.read_csv(file, sep=";", encoding="latin-1")
    # Remplacer les valeurs vides par "Non définie" dans le dataframe
    df = df.fillna("Non définie")
    #df["Species"].fillna("None", inplace=True)

    return df


# -------------------------------------------------------------------------------
# Function to display the initial dataframe ==> Not USED
def print_data(df):
    data_printed = st.checkbox("Cliquez pour voir votre table de données")
    if data_printed:
        st.dataframe(df, use_container_width=True)


# -------------------------------------------------------------------------------
# Function to display the work dataframe ==> will disappear in the final version
def print_work_data(df):
    st.write(df)


# -------------------------------------------------------------------------------
# Change the type of the data of the dataframe
def manage_type_data(df):
    df['Species'] = df['Species'].astype('string')
    df['Category'] = df['Category'].astype('string')
    df["Time"] = pd.to_datetime(df["Time"])
    return df


# -------------------------------------------------------------------------------
# Create details columns for the time
def detail_date(df):
    df["Year"] = df["Time"].dt.year
    df["Month"] = df["Time"].dt.month
    df["Day"] = df["Time"].dt.day
    df["Hour"] = df["Time"].dt.hour
    df["Minute"] = df["Time"].dt.minute
    return df


# -------------------------------------------------------------------------------
# CSS
st.markdown(
    """
    <style>
    /* Styles CSS pour le titre d'une page */
    .page-title {
        text-align: center;
        font-weight: bold;
        font-size: xxx-large;
        color: #ffffff;
        backdrop-filter: blur(1px);
    } 

    /* Styles CSS pour les conteneurs de texte */
    .txt-container-flower {
        background-color: #BFC4BC;
        color: #000000;
        border-radius: 25% 10% / 5% 20%;
        padding: 5%;
        margin: 3%;
        text-align: justify;
    }

    /* Style CSS pour box container des différents insectes */
    .Box-container {
        background-color: #BFC4BC;
        color: #000000;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
    }

    /* Style CSS pour les items de la grille de la liste des différents insectes */
    .custom-grid-list-item {
        white-space: nowrap;
        margin: 3%;
    } 

    h3 {
        color: #ffffff;
        backdrop-filter: blur(1px);
    }

    h4 {
        color: #ffffff;
        backdrop-filter: blur(1px);
    }

    h5 {
        color: #000000;
    }

    /* Solid border */
    hr.solid {
        border-top: 3px solid #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True
)

hide_menu = """
<style>
#MainMenu{
    visibility:hidden;
    }
footer{
    visibility:hidden;
}
footer:after{
    content : 'Copyright @2023: GodSaveTheBugs';
    display:block;
    position:relative;
    color:white;
    visibility: visible;
    padding:5px;
    top:2px;
}
header{
    visibility:hidden;
}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)



# -------------------------------------------------------------------------------
# Compare two dataframes
def compare_dataframes(df1, df2):
    # Vérifier si les noms des colonnes sont identiques
    if set(df1.columns) != set(df2.columns):
        return False

    # Vérifier si les données dans chaque colonne sont identiques
    for col in df1.columns:
        if not df1[col].equals(df2[col]):
            return False

    # Vérification réussie, les DataFrames sont identiques
    return True


# -------------------------------------------------------------------------------
# Métric sur l'indice de biodiversité permettant de savoir le nombre d'insectes différents détectés
def personnamize_metric(number, text):            
    st.markdown(
        """
        <style>
        [data-testid="stMetricLabel"] {
            font-size: 2em;
        }
        [data-testid="stMetricValue"] {
            font-size: 6em;
            color: #ffffff;
            backdrop-filter: blur(1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<h5 style='font-weight:bold; color: #ffffff; backdrop-filter: blur(1px);'>{text}</h5>", unsafe_allow_html=True)
    st.metric(label="", value=number)

# -------------------------------------------------------------------------------
# Home Page
def Home_Page():
    colmargrleft, colmain, colmargeright = st.columns([1, 10, 1], gap="small")

    with colmain:
        st.markdown(f'''
            <h1 class=page-title>God Saves The Bugs</h1>
            <h3 style=>Bienvenue dans notre application de mesure de la biodiversité !</h3>
            <p class=txt-container-flower>Notre objectif est de vous aider à évaluer et à comprendre la biodiversité d'un lieu spécifique en utilisant des données préalablement collectées et analysées par notre algorithme de machine learning. La biodiversité est essentielle pour maintenir l'équilibre écologique de notre planète, et il est crucial de surveiller et de comprendre les différentes espèces qui peuplent un environnement donné. Grâce à notre algorithme de machine learning, nous avons été en mesure d'extraire des informations précieuses à partir de vastes ensembles de données sur la biodiversité. Notre application vous permettra d'explorer ces données et de visualiser les espèces présentes dans le lieu de votre choix. Nous croyons fermement que la technologie et l'intelligence artificielle peuvent jouer un rôle essentiel dans la conservation et la préservation de notre biodiversité. En utilisant les avancées de l'apprentissage automatique, nous sommes en mesure de fournir des données fiables et des visualisations interactives pour vous aider à mieux comprendre et apprécier la richesse biologique d'un lieu. Nous espérons que notre application vous aidera à sensibiliser à l'importance de la biodiversité et à prendre des décisions éclairées pour sa protection. Profitez de l'exploration de notre application et découvrez la diversité étonnante de la vie qui peuple notre planète.</p>
        ''', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1], gap="medium")

        with col1:
            # Gérer le taille de la box pour être raccord avec la box de l'indice
            st.markdown(f'''
                <p class=txt-container-flower>L'indice de biodiversité mesure la diversité et la richesse des espèces dans un environnement donné. Il est basé sur le nombre d'espèces présentes, offrant un aperçu de la biodiversité de l'écosystème étudié. Une plus grande diversité d'espèces est généralement associée à un écosystème plus sain et résilient, offrant une variété de rôles écologiques et de services écosystémiques. Mesurer cet indice permet d'évaluer la santé de l'écosystème, son niveau de stabilité et sa capacité à faire face aux changements environnementaux.</p>
            ''', unsafe_allow_html=True)

        with col2:
            species_number = st.session_state.initial_data[st.session_state.initial_data["Species"] != "Non définie"]["Species"].nunique()
            # Indice de biodiversité sur le nombre d'espèces différentes dans l'environnment
            personnamize_metric(species_number, "Indice de biodiversité")

        st.markdown("<hr class=solid>", unsafe_allow_html=True)

        st.session_state.view_data = st.checkbox("Cliquez pour voir votre table de données")
        if st.session_state.view_data:
            st.dataframe(st.session_state.initial_data, use_container_width=True)


# -------------------------------------------------------------------------------
# Fonction pour récupérer les noms des espèces du dataframe (retire la valeur "nan"). Return une liste
def get_unique_in_col(col_name):
    list_unique_name = st.session_state.initial_data[col_name].unique().tolist()
    list_unique_name = [name for name in list_unique_name if pd.notna(name) and name != "Non définie"]
    return list_unique_name


# -------------------------------------------------------------------------------
# Fonction créer un pie chart interactif
# ATTENTION : VOIR SI POSSIBILITE DE GERER LES COULEURS DU PIECHART
def create_pie_chart(column_name, background):
    # Compter le nombre d'occurrences de chaque nom
    value_counts = st.session_state.initial_data[column_name].value_counts(dropna=False)

    # Créer un DataFrame avec les noms et les occurrences
    data = pd.DataFrame({'Espèces': value_counts.index, 'Occurrences': value_counts.values})

    # Renommer les valeurs nulles en "non-ientifié"
    data['Espèces'].fillna('Non définie', inplace=True)

    # Créer le pie chart interactif avec Plotly Express en utilisant une échelle de couleurs verte
    fig = go.Figure(data=[go.Pie(labels=data['Espèces'], values=data['Occurrences'],
                                 marker=dict(colors=px.colors.sequential.Greens),
                                 textposition='inside', textinfo='percent+label')])
    
    fig.update_layout(title='Répartition des espèces observées :')

    # Ajouter une bordure autour de chaque carré de légende
    for i, trace in enumerate(fig.data):
        if 'marker' in trace:
            fig.data[i].marker.line = dict(color='#BFC4BC', width=1)

    # Modifier la couleur de fond et du texte du graphique
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Définir la couleur de fond sur transparent
        paper_bgcolor='rgba(0, 0, 0, 0)', # Définir la couleur de fond du papier (conteneur du graphique)
        legend=dict(font=dict(color='white'), bgcolor='rgba(0, 0, 0, 0)',
                    borderwidth=1),  # Ajouter une bordure autour de la légende
        title=dict(font=dict(color='white'))
    )

    if background == True:
        fig.update_layout(
        paper_bgcolor='#BFC4BC',
        )

    return fig


# -------------------------------------------------------------------------------
# Définir le nombre de colonnes pour affichage 
def nb_col(list):
    item_list = len(list)
    num_col = round(item_list / 4)
    return num_col


# -------------------------------------------------------------------------------
# Créer un bar chart du pourcentage d'espèces en fonction de la catégorie
def bar_chart(color_map, title_space, background):
    # Filtrer les catégories pour exclure "Non définie"
    categories = st.session_state.initial_data['Category'].unique()
    categories = [category for category in categories if category != "Non définie"]

    # Calculer le pourcentage de chaque catégorie dans l'environnement
    category_percentage = st.session_state.initial_data.loc[
        st.session_state.initial_data['Category'].isin(categories)
    ]['Category'].value_counts(normalize=True) * 100

    # Créer une liste pour stocker les valeurs de pourcentage formatées
    category_percentage_str = []

    # Convertir les valeurs de pourcentage en chaînes de caractères et les ajouter à la liste
    for value in category_percentage.values.round(2):
        category_percentage_str.append(f"{value:.2f}%")

    # Créer une figure Plotly
    fig = go.Figure()

    # Ajouter les barres à la figure
    fig.add_trace(go.Bar(
        x=category_percentage.index,
        y=category_percentage.values,
        marker=dict(
            color=[color_map[category] for category in category_percentage.index]
        ),
        text=category_percentage_str,
        textposition='auto',
        hovertemplate="Pourcentage dans la catégorie: %{text}",
        textfont=dict(color='white')
    ))

    # Mettre en forme la figure
    fig.update_layout(
        xaxis=dict(title='Catégorie', title_font=dict(color='white'),tickfont=dict(color='white'),showgrid=False),
        yaxis=dict(title='Pourcentage', title_font=dict(color='white'),tickfont=dict(color='white'),showgrid=False),
        showlegend=False,
        title="Pourcentage des catégories dans l'environnement",
        title_x=title_space,
        title_font=dict(size=18, color='white'),
    )

    # Modifier la couleur de fond et du texte du graphique
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)', # Définir la couleur de fond sur transparent
        paper_bgcolor='rgba(0, 0, 0, 0)',# Définir la couleur de fond du papier (conteneur du graphique)
        legend={'font': {'color': 'white'}}
    ) 

    if background == True:
        fig.update_layout(
        paper_bgcolor='#BFC4BC',
        )

    return fig


# -------------------------------------------------------------------------------
# Créer un line chart du nombre d'espèces par heure
def esp_time():
    st.session_state.initial_data["Day"] = st.session_state.initial_data["Time"].dt.hour
    st.session_state.initial_data["Month"] = st.session_state.initial_data["Time"].dt.day
    st.session_state.initial_data["Year"] = st.session_state.initial_data["Time"].dt.month

    # ComboBox for frequency selection
    options = ["Année", "Mois", "Journée"]
    selectionFrequency = st.radio("Sélectionner la période d'analyse :", options)

    # ComboBox for mode selection
    options = ["Catégories entre elles", "Espèces entre elles", "Espèces au sein d'une catégorie"]
    selectionMode = st.radio("Sélectionner le mode d'analyse :", options)

    mode = ''
    graph = False

    if selectionMode == "Catégories entre elles":
        mode = "Category"
        # 1. Create a variable to store categories.
        if not 'categories' in st.session_state:
            st.session_state.categories = []

        # 2. Prompt the user in the form
        st.session_state.categories = get_unique_in_col("Category")

        # 3. Display the contents of the list
        selected_categories = st.multiselect(
            'Sélectionner les catégories souhaitées :',
            st.session_state.categories)

        if selected_categories:
            selected_species_data = st.session_state.initial_data[st.session_state.initial_data["Category"].isin(selected_categories)]
            df = pd.DataFrame(selected_species_data, columns=["Species", "Category", "Time", "Day", "Month", "Year"])
            graph = True

    elif selectionMode == "Espèces entre elles":
        mode = "Species"
        # 1. Create a variable to store species.
        if not 'species' in st.session_state:
            st.session_state.species = []

        # 2. Prompt the user in the form
        st.session_state.species = get_unique_in_col("Species")

        # 3. Display the contents of the list
        selected_species = st.multiselect(
            'Sélectionner les espèces souhaitées :',
            st.session_state.species)

        if selected_species:
            selected_species_data = st.session_state.initial_data[st.session_state.initial_data["Species"].isin(selected_species)]
            df = pd.DataFrame(selected_species_data, columns=["Species", "Category", "Time", "Day", "Month", "Year"])
            graph = True

    elif selectionMode == "Espèces au sein d'une catégorie":
        mode = "Species"
        # 1. Create a variable to store categories.
        if not 'categories' in st.session_state:
            st.session_state.categories = []

        # 2. Prompt the user in the form
        st.session_state.categories = get_unique_in_col("Category")

        # 3. Display the contents of the list
        selected_cat = st.selectbox("Sélectionner la catégorie d'insectes souhaitée :", st.session_state.categories)

        if selected_cat:
            selected_cat_data = st.session_state.initial_data[st.session_state.initial_data["Category"] == selected_cat]
            df = pd.DataFrame(selected_cat_data, columns=["Species", "Category", "Time", "Day", "Month", "Year"])
            graph = True


    if graph:
        if selectionFrequency == "Année":
            filtered_data = df.pivot_table(index='Year', columns=mode, aggfunc='size', fill_value=0)
            scale = "Mois"

        elif selectionFrequency == "Mois":
            filtered_data = df.pivot_table(index='Month', columns=mode, aggfunc='size', fill_value=0)
            scale = "Jours"

        elif selectionFrequency == "Journée":
            filtered_data = df.pivot_table(index='Day', columns=mode, aggfunc='size', fill_value=0)
            scale = "Heures"

        # Créer le line chart avec Plotly
        fig = px.line(filtered_data, color_discrete_sequence=px.colors.qualitative.Dark24)

        # Mettre en forme la figure
        fig.update_layout(
            autosize=True,
            xaxis=dict(title=f'{scale}', title_font=dict(color='white'),tickfont=dict(color='white')),
            yaxis=dict(title="Nombre d'individus", title_font=dict(color='white'),tickfont=dict(color='white')),
            title=f"Présence des insectes en fonction du temps",
            title_x=0.25,
            title_font=dict(size=18, color='white'),
            legend=dict(title=dict(font=dict(color='white')))
        )

        # Changer la couleur du fond du graphique en gris
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Définir la couleur de fond sur transparent
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Définir la couleur de fond du papier (conteneur du graphique)
            legend={'font': {'color': 'white'}}
        ) 
        # Modifier l'échelle de l'axe des abscisses pour un espacement précis
        fig.update_xaxes(dtick='M1')

        # Display the line chart
        st.plotly_chart(fig, use_container_width=True)



# -------------------------------------------------------------------------------
# Page sur la répartition du nombre d'individus dans l'environnement d'étude
def Ind_in_env():
    colmargrleft, colmain, colmargeright = st.columns([1, 10, 1], gap="medium")

    with colmain:
        list_of_species = get_unique_in_col("Species")

        st.markdown(f'''
            <h1 class=page-title>Vos Individus Répertoriés</h1>
        ''', unsafe_allow_html=True)

        nb_cols = nb_col(list_of_species)

        # st.write(list_of_species)

        # Code HTML pour le style et le script
        st.markdown(f'''
        <style>
            .custom-list {{
                display: grid; 
                grid-template-columns: repeat({nb_cols},1fr);
                overflow-x: scroll;
                margin: 0;
                scrollbar-gutter: stable;
                column-gap: 3%;
            }}
        </style>

        <section class="Box-container">
            <h5>Les espèces répertoriées dans votre environnement sont :</h5>
            <ul class="custom-list">
                {"".join(f'<li class="custom-grid-list-item">{item}</li>' for item in list_of_species)}
            </ul>
        </section>
        ''', unsafe_allow_html=True)

        fig = create_pie_chart("Species", False)

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.write("Vous voulez en apprendre davantage sur la répartition des insectes ?")

        description = st.button("En savoir plus...")

        if description:
            st.markdown(f'''
                <p class=txt-container-flower>La répartition des insectes dans un environnement est essentielle pour plusieurs raisons. Premièrement, elle constitue un indicateur important de la biodiversité locale, reflétant l'état de l'écosystème. Deuxièmement, les insectes jouent un rôle clé en fournissant des services écosystémiques tels que la pollinisation et la régulation des ravageurs. Troisièmement, ils contribuent à maintenir l'équilibre écologique en régulant les populations d'autres organismes et en étant une source de nourriture pour de nombreux animaux. Enfin, la répartition des insectes peut servir d'indicateur précoce de changements environnementaux, fournissant des informations précieuses pour la conservation et la gestion des écosystèmes.</p>
            ''', unsafe_allow_html=True)

# -------------------------------------------------------------------------------
# Fonction pour avoir un df avec chaque espèce ainsi que leur catégorie, leur nombre et le pourcentage que ca représente dans l'environnement
def df_occurences_species(df):
    # Compter le nombre d'occurrences de chaque espèce
    nombre_occurrences = df['Species'].value_counts().reset_index()
    # Supprimer les lignes où la valeur de la deuxième colonne est "non défini"
    nombre_occurrences = nombre_occurrences[nombre_occurrences.iloc[:, 0] != "Non définie"]
    nombre_occurrences.columns = ['Species', 'Nombre d occurrences']

    # Calculer le pourcentage de chaque espèce par rapport au total
    total = nombre_occurrences['Nombre d occurrences'].sum()
    nombre_occurrences['Pourcentage'] = (nombre_occurrences['Nombre d occurrences'] / total) * 100

    # Récupérer les catégories uniques sans la valeur spécifique
    categories = df['Category'].loc[df['Category'] != 'Non définie'].unique()

    # Trier le DataFrame par ordre alphabétique des espèces
    nombre_occurrences = nombre_occurrences.sort_values('Species')

    # Effectuer une jointure entre nombre_occurrences et df pour obtenir les catégories correspondantes
    nombre_occurrences = nombre_occurrences.merge(df[['Species', 'Category']].drop_duplicates(), on='Species', how='left')

    return nombre_occurrences, categories

# -------------------------------------------------------------------------------
# Visualisation des ronds
def visu_ronds(df):

    nombre_occurrences, categories = df_occurences_species(df)

    # Définir les couleurs pour les catégories
    #color_map = {category: f'#{hash(category) % 0xffffff:06x}' for category in categories}
    color_map = {category: color for category, color in zip(categories, px.colors.qualitative.Plotly)}

    # Précalculer les indices pour les coordonnées x et y des cercles
    indices = np.arange(len(nombre_occurrences))
    row_count = int(np.ceil(len(nombre_occurrences) / 4))
    x_indices = (indices % 4)
    y_indices = row_count - 1 - np.floor(indices / 4)

    # Créer une figure Plotly
    fig = go.Figure()

    # Ajouter des cercles à la figure
    circle_size = nombre_occurrences['Nombre d occurrences'] * 100
    max_circle_size = circle_size.max()
    circle_area = np.pi * (circle_size / max_circle_size) ** 2
    fig.add_trace(go.Scatter(
        x=x_indices,
        y=y_indices,
        mode='markers',
        marker=dict(
            size=circle_area,
            sizemode='area',
            sizeref=2 * max(circle_area) / (100 ** 2),
            color=[color_map[category] for category in nombre_occurrences['Category']],
            opacity=0.7
        ),
        hovertemplate="Espèce: %{text}<br>Nombre d'individus: %{customdata[0]}<br>Pourcentage dans l'environnement: %{customdata[1]:.2f}%",
        text=nombre_occurrences['Species'],
        customdata=nombre_occurrences[['Nombre d occurrences', 'Pourcentage']],
        showlegend=False
    ))

    # Mettre en forme la disposition de la figure
    fig.update_layout(
        autosize = True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=None, # ==> changer la couleur.
        title="Pourcentage des espèces dans l'environnement",
        title_x=0.15,
        title_font=dict(size=18, color='white')
    )

    # Modifier la couleur de fond et du texte du graphique
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Définir la couleur de fond sur transparent
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Définir la couleur de fond du papier (conteneur du graphique)
        legend={'font': {'color': 'white'}}
    ) 

    # Ajouter le nom de l'espèce en dessous des cercles avec décalage variable
    for i, row in nombre_occurrences.iterrows():
        circle_radius = np.sqrt(circle_area[i] / np.pi)  # Calculer le rayon du cercle correspondant
        fig.add_annotation(
            x=x_indices[i],
            y=y_indices[i] - (circle_radius/2),
            text=row['Species'],
            showarrow=False,
            font=dict(size=12, color='white'),
            xshift=0,
            yshift=0
        )
    # Ajouter le pourcentage au milieu des cercles
        fig.add_annotation(
            x=x_indices[i],
            y=y_indices[i],
            text=f"{round(row['Pourcentage'])}%",
            showarrow=False,
            font=dict(size=10, color='white'),
            xshift=0,
            yshift=0
        )

    # Ajouter une trace pour chaque catégorie avec un seul marqueur pour créer la légende
    for category in categories:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=10,
                color=color_map[category]
            ),
            name=category,
            showlegend=True
        ))

    # Ajouter le titre à la figure
    fig.update_layout(
        legend=dict(
            title=dict(
                text='Catégorie',
                font=dict(color='white')
            ),
            traceorder='normal',
            font=dict(size=12),
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='right',
            x=1.25,
            itemclick=False
        )
    )

    # Vérifier si la visu contient une seule ligne
    if row_count == 1:
        fig.update_layout(
            yaxis_range=[-1, 1]
        )

    return fig, color_map

# -------------------------------------------------------------------------------
# permet d'afficher notre logo sur la sidebar
@st.cache
def add_logo(logo_path, width, height):
    #Lire et retourner l'image
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo
# -------------------------------------------------------------------------------
# récupérer les catégories avec plusieurs espèces
def category_with_multiple_species():
    df_copy = st.session_state.initial_data.copy()

    # Supprimer les doublons basés sur la paire ["Species", "Category"]
    df_copy.drop_duplicates(subset=["Species", "Category"], inplace=True)

    # Identifier les doublons de la colonne "Category"
    duplicates = df_copy[df_copy.duplicated(subset=["Category"], keep=False)]

    # Extraire les catégories avec des doublons
    categories_with_duplicates = duplicates["Category"].unique().tolist()

    return categories_with_duplicates

# -------------------------------------------------------------------------------
# Page sur l'étude du pourcentage d'espèces différentes par catégorie
def Per_by_Category():
    colmargrleft, colmain, colmargeright = st.columns([1, 10, 1], gap="medium")

    with colmain:
        st.markdown(f'''
                <h1 class=page-title>Pourcentage d'espèces par catégorie</h1>
            ''', unsafe_allow_html=True)
        list_of_category = get_unique_in_col("Category")
        # st.write(list_of_category)
        nb_cols = nb_col(list_of_category)

        # Code HTML pour le style et le script
        st.markdown(f'''
        <style>
            .custom-list {{
                display: grid; 
                grid-template-columns: repeat({nb_cols},1fr);
                overflow-x: scroll;
                margin: 0;
                scrollbar-gutter: stable;
                column-gap: 3%;
            }}
        </style>

        <section class="Box-container">
            <h5>Les catégories présentes dans votre environnement sont :</h5>
            <ul class="custom-list">
                {"".join(f'<li class="custom-grid-list-item">{item}</li>' for item in list_of_category)}
            </ul>
        </section>
        ''', unsafe_allow_html=True)

        categories = category_with_multiple_species()

        categories.insert(0, "Toutes")

        filtering_option = st.selectbox("Quelle catégorie voulez-vous observer ? (non définies exclues)", categories)

        if filtering_option == "Toutes":
            fig, color_map = visu_ronds(st.session_state.initial_data)
            st.plotly_chart(fig, use_container_width=True)
            fig = bar_chart(color_map, 0.25, False)
            # Afficher la figure dans Streamlit
            st.plotly_chart(fig, use_container_width=True)
        else:
            filtered_data = st.session_state.initial_data[st.session_state.initial_data['Category'] == filtering_option]
            fig, color_map = visu_ronds(filtered_data)
            st.plotly_chart(fig, use_container_width=True)
        

        st.markdown("<hr class=solid>", unsafe_allow_html=True)

        information = st.selectbox(
            "L'importance d'avoir plusieurs catégories d'insectes dans un environnement donné :",
            ('Compréhension de la biodiversité', "Indicateur de l'état de l'écosystème", 'Conservation'))
        if information == 'Compréhension de la biodiversité':
            st.markdown(f'''
                <p class=txt-container-flower>Le pourcentage d'espèces par catégorie d'insectes permet de comprendre la répartition des différentes familles, genres ou ordres d'insectes dans un écosystème donné. Cela donne une idée de la diversité taxonomique des insectes présents.</p>
            ''', unsafe_allow_html=True)
        elif information == "Indicateur de l'état de l'écosystème":
            st.markdown(f'''
                <p class=txt-container-flower>Certains groupes d'insectes peuvent être considérés comme des indicateurs écologiques de l'état de santé d'un écosystème. Par exemple, les papillons peuvent être utilisés pour évaluer la qualité des habitats et les effets des perturbations environnementales.</p>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <p class=txt-container-flower>Certains groupes d'insectes peuvent être considérés comme étant plus menacés ou vulnérables que d'autres. En évaluant le pourcentage d'espèces par catégorie, on peut identifier les groupes nécessitant une attention particulière en termes de conservation et de gestion des habitats.</p>
            ''', unsafe_allow_html=True)


# -------------------------------------------------------------------------------
# Page d'analyse de données en fonction du temps (by year, by month, by day, by hour, by minute)
def Data_and_Time():
    colmargrleft, colmain, colmargeright = st.columns([1, 15, 1], gap="medium")

    with colmain:
        st.markdown(f'''
                <h1 class=page-title>Présence des insectes en fonction du temps</h1>
            ''', unsafe_allow_html=True)

        st.session_state.initial_data["Time"] = pd.to_datetime(st.session_state.initial_data["Time"])

        esp_time()


# -------------------------------------------------------------------------------
# Page sur la présentation de l'équipe
@st.cache_data
def Page_team():
    st.markdown(f'''
            <h1 class=page-title>Qui sommes-nous ?</h1>
        ''', unsafe_allow_html=True)

    team_zip = 'team.zip'
    if 'team_images' not in st.session_state:
        with ZipFile(team_zip, 'r') as zip_ref:
            image_files = zip_ref.namelist()
            images = [Image.open(zip_ref.open(image_file)) for image_file in image_files]
        st.session_state.team_images = images

    N_PICS = 8
    n_rows = int(1 + N_PICS // 4)
    rows = [st.columns(4) for _ in range(n_rows)]
    cols = [column for row in rows for column in row]

    image_descriptions = [
        "Anne-Julie HOTTIN - Data analyst",
        "Victoria STASIK - Artificial Intelligence engineer",
        "Théo MASSON - Data analyst",
        "Cédric SONG - Data analyst",
        "Mike LEVELEUX - Artificial Intelligence engineer",
        "Justine BOILLOT - Project Manager - Software engineer",
        "Justine WANG - IRV engineer",
        "Nouhayla CHERRABI - Artificial Intelligence engineer"
    ]

    for col, (team_image, description) in zip(cols, zip(st.session_state.team_images, image_descriptions)):
        col.image(team_image)
        col.markdown(f"<p style='text-align:center;'>{description}</p>", unsafe_allow_html=True)



# -------------------------------------------------------------------------------
# Custom class for the PDF report
class PDF(FPDF,HTMLMixin):
    def header(self):
        # Add header content here
        self.image("logo.png", x=5, y=0, w=25)
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Mon rapport sur la biodiversité", align="C")

    def footer(self):
        # Add footer content here
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, "Mon rapport sur la biodiversité - " + date.today().strftime("%d/%m/%Y"), 0, 0, "C")

    def chapter_title(self, title):
            # Code pour afficher le titre du chapitre dans le PDF
            self.set_font("Arial", "B", 26)
            self.cell(70, 10, title, ln=True, align="C")
            #self.ln(10)

    def chapter_content(self, content):
        # Code pour afficher le contenu du chapitre dans le PDF
        self.set_font("Arial", "", 12)
        self.multi_cell(70, 10, content, border=0, align='C')
        #self.ln(5)

    def table_content(self, content):
        # Code pour afficher le contenu de la table dans le PDF
        self.set_font("Arial", "", 12)
        self.multi_cell(100, 10, content, border=0, align='C')
        #self.ln(5)

    def chapter_small_content(self, content):
        # Code pour afficher le contenu du chapitre dans le PDF
        self.set_font("Arial", "", 8)
        self.multi_cell(70, 5, content, border=0, align='C')
        #self.ln(5)

    def write_html_table(self, html_content):
        table_rows = html_content.split('\n')  # Séparer les lignes du contenu du tableau
        for row in table_rows:
            cells = row.split('|')  # Séparer les cellules de chaque ligne
            for cell in cells:
                self.cell(40, 10, cell.strip(), border=1)  # Afficher chaque cellule avec une bordure
            self.ln()  # Aller à la ligne suivante après chaque ligne du tableau

    def add_table(self, table_content):
        self.set_font("Arial", size=10)
        # Format the table content
        headers = table_content.columns.tolist()
        rows = table_content.values.tolist()
        # Get the maximum width for each column
        col_widths = [1.1 * max(self.get_string_width(str(header)), max(self.get_string_width(str(row[i])) for row in rows)) for i, header in enumerate(headers)]
        # Write the table headers
        for header, width in zip(headers, col_widths):
            self.cell(width, 10, str(header), 1, 0, "C")
        self.ln()
        # Write the table rows
        for row in rows:
            for value, width in zip(row, col_widths):
                self.cell(width, 10, str(value), 1, 0, "C")
            self.ln()
        self.ln(5)


# -------------------------------------------------------------------------------
# Function to generate the PDF report ==> A changer en fonction de rapport défini
def generate_report(color_map):
    # Create a new PDF document
    pdf = PDF()

    # Génération du rapport PDF
    pdf.add_page()

    # Set up the document
    #pdf.set_auto_page_break(auto=True, margin=15)

    # Add content to the report
    pdf.set_font("Arial", size=12)
    pdf.ln(15)

    # Ajout du pie chart
    pie_chart = create_pie_chart("Species", True)
    # Sauvegarde du pie chart dans un fichier image temporaire
    pie_chart_filename = "pie_chart.png"
    pie_chart.write_image(pie_chart_filename)
    # Ajout de l'image dans le rapport PDF
    pdf.image(pie_chart_filename, x=5, y=pdf.get_y(), w=115)
    pdf.ln(10)

    # Ajouter l'indice du nombre d'espèces
    pdf.rect(135, 25, 70, 38, 'D')

    # Définir les paramètres pour le texte à l'intérieur du rectangle
    text_x = 135  # Coordonnée X du coin supérieur gauche du rectangle
    text_y = 25  # Coordonnée Y du coin supérieur gauche du rectangle
    pdf.set_xy(text_x, text_y)  # Définir les coordonnées de départ du texte
    pdf.chapter_content("Indice de biodiversité")

    text_x = 135  # Coordonnée X du coin supérieur gauche du rectangle
    text_y = 36  # Coordonnée Y du coin supérieur gauche du rectangle
    species_number = st.session_state.initial_data[st.session_state.initial_data["Species"] != "Non définie"]["Species"].nunique()
    pdf.set_xy(text_x, text_y)  # Définir les coordonnées de départ du texte
    pdf.chapter_title(f"{species_number}")

    text_x = 135  # Coordonnée X du coin supérieur gauche du rectangle
    text_y = 49  # Coordonnée Y du coin supérieur gauche du rectangle
    pdf.set_xy(text_x, text_y)  # Définir les coordonnées de départ du texte
    pdf.chapter_small_content("Indice sur le nombre d'espèces reconnues dans l'environnement.")

    # Ajouter l'indice du nombre d'individus
    pdf.rect(135, 68, 70, 43, 'D')

    # Définir les paramètres pour le texte à l'intérieur du rectangle
    text_x = 135  # Coordonnée X du coin supérieur gauche du rectangle
    text_y = 68  # Coordonnée Y du coin supérieur gauche du rectangle
    pdf.set_xy(text_x, text_y)  # Définir les coordonnées de départ du texte
    pdf.chapter_content("Nombre d'individus")

    text_x = 135  # Coordonnée X du coin supérieur gauche du rectangle
    text_y = 79  # Coordonnée Y du coin supérieur gauche du rectangle
    df_copy = st.session_state.initial_data.copy()
    # Exclure les lignes où la colonne "Species" est "Non définie"
    df_filtered = df_copy[df_copy['Species'] != 'Non définie']
    # Calculer le nombre de lignes dans le DataFrame pour avoir le nombre d'individus detectés
    nb_indiv = len(df_filtered)
    pdf.set_xy(text_x, text_y)  # Définir les coordonnées de départ du texte
    pdf.chapter_title(f"{nb_indiv}")

    text_x = 135  # Coordonnée X du coin supérieur gauche du rectangle
    text_y = 90  # Coordonnée Y du coin supérieur gauche du rectangle
    pdf.set_xy(text_x, text_y)  # Définir les coordonnées de départ du texte
    pdf.chapter_small_content("Indice sur le nombre d'individus reconnus dans l'environnement.")

    text_x = 135  # Coordonnée X du coin supérieur gauche du rectangle
    text_y = 100  # Coordonnée Y du coin supérieur gauche du rectangle
    # Obtenir la plage de la colonne Time
    df_filtered['Time'] = pd.to_datetime(df_filtered['Time'],dayfirst=True)
    plage_time = (df_filtered['Time'].min(), df_filtered['Time'].max())
    # Phrase descriptive
    phrase_info = f"Vos données sont basées sur des dates du {plage_time[0].strftime('%d/%m/%Y')} au {plage_time[1].strftime('%d/%m/%Y')}"
    pdf.set_xy(text_x, text_y)  # Définir les coordonnées de départ du texte
    pdf.chapter_small_content(phrase_info)

    df_species_occ_info, categories = df_occurences_species(st.session_state.initial_data)
    # Renommer les colonnes
    nouvelles_colonnes = {
       'Species': 'Espèce',
        'Nombre d occurrences': 'Nombre d\'individus',
        'Pourcentage': '% dans l\'environnement',
        'Category': 'Catégorie'
    }
    df_species_occ_info = df_species_occ_info.rename(columns=nouvelles_colonnes)
    df_species_occ_info['% dans l\'environnement'] = df_species_occ_info['% dans l\'environnement'].round(1)
    df_species_occ_info['% dans l\'environnement'] = df_species_occ_info['% dans l\'environnement'].apply(lambda x: f"{x}%")
    df_species_occ_info['Nombre d\'individus'] = df_species_occ_info['Nombre d\'individus'].apply(lambda x: f"{x} individus")

    # Modifier l'ordre des colonnes
    nouvel_ordre = ['Espèce', 'Catégorie', 'Nombre d\'individus', '% dans l\'environnement']
    df_species_occ_info = df_species_occ_info.reindex(columns=nouvel_ordre)

    # Trier les lignes par ordre alphabétique des catégories
    df_trie = df_species_occ_info.sort_values(by='Catégorie')

    # pdf.rect(5, 115, 115, 100, 'D')

    text_x = 5  # Coordonnée X du coin supérieur gauche 
    text_y = 115  # Coordonnée Y du coin supérieur gauche 
    pdf.set_xy(text_x, text_y)  # Définir les coordonnées de départ du texte
    pdf.table_content("Table d'information de votre environnement")


    # Convertir le DataFrame en un tableau formaté
    table_content = df_trie

    # Ajouter le contenu du tableau au rapport PDF
    pdf.add_table(table_content)
    pdf.ln(10)

    # Ajout du bar chart
    barchart = bar_chart(color_map, 0, True)
    # Sauvegarde du bar chart dans un fichier image temporaire
    barchart_filename = "barchart.png"
    barchart.write_image(barchart_filename)
    # Ajout de l'image dans le rapport PDF
    #graph_x = 125  # Coordonnée X du coin supérieur gauche du graphique
    graph_y = pdf.get_y() - 12 # Coordonnée Y du coin supérieur gauche du graphique
    graph_width = 21 * len(categories)  # Largeur du graphique
    pdf.image(barchart_filename, x=pdf.get_x(), y=graph_y, w=graph_width)
    pdf.ln(10)
    
    # Suppression du fichier image temporaire du pie chart
    os.remove(pie_chart_filename)
    os.remove(barchart_filename)


    # Get the current date
    today = date.today()
    formatted_date = today.strftime("%Y-%m-%d")

    # Save the PDF document with the date in the filename
    report_filename = f"report_{formatted_date}.pdf"
    pdf.output(report_filename)

    return report_filename


# -------------------------------------------------------------------------------
# Rapport général
def report():
    st.markdown(f'''
        <h1 class=page-title>Analyse de la biodiversité</h1>
    ''', unsafe_allow_html=True)

    st.markdown("<hr class=solid>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig = create_pie_chart("Species", False)
        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)
    with col2:

        colb1, colb2 = st.columns([1, 1], gap="medium")

        with colb1:
            st.write("Indice sur le nombre d'espèces reconnues dans l'environnement.")
            species_number = st.session_state.initial_data[st.session_state.initial_data["Species"] != "Non définie"]["Species"].nunique()
            # Indice de biodiversité sur le nombre d'espèces différentes dans l'environnment
            personnamize_metric(species_number, "Indice de biodiversité")

        with colb2:
            st.write("Indice sur le nombre d'individus reconnus dans l'environnement.")

            df_copy = st.session_state.initial_data.copy()

            # Exclure les lignes où la colonne "Species" est "Non définie"
            df_filtered = df_copy[df_copy['Species'] != 'Non définie']

            # Calculer le nombre de lignes dans le DataFrame pour avoir le nombre d'individus detectés
            nb_indiv = len(df_filtered)

            # Obtenir la plage de la colonne Time
            df_filtered['Time'] = pd.to_datetime(df_filtered['Time'],dayfirst=True)
            plage_time = (df_filtered['Time'].min(), df_filtered['Time'].max())

            # Indice de biodiversité sur le nombre d'espèces différentes dans l'environnment
            personnamize_metric(nb_indiv, "Nombre d'individus")

            # Phrase descriptive
            phrase_info = f"Vos données sont basées sur des dates du {plage_time[0].strftime('%d/%m/%Y')} au {plage_time[1].strftime('%d/%m/%Y')}"

            # Affichage de la phrase descriptive
            st.write(phrase_info)

    st.markdown("<hr class=solid>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")
    fig, color_map = visu_ronds(st.session_state.initial_data)
    with col1:
        df_species_occ_info, categories = df_occurences_species(st.session_state.initial_data)
        # Renommer les colonnes
        nouvelles_colonnes = {
            'Species': 'Espèce',
            'Nombre d occurrences': 'Nombre d\'individus',
            'Pourcentage': '% dans l\'environnement',
            'Category': 'Catégorie'
        }
        df_species_occ_info = df_species_occ_info.rename(columns=nouvelles_colonnes)
        df_species_occ_info['% dans l\'environnement'] = df_species_occ_info['% dans l\'environnement'].round(1)
        df_species_occ_info['% dans l\'environnement'] = df_species_occ_info['% dans l\'environnement'].apply(lambda x: f"{x}%")
        df_species_occ_info['Nombre d\'individus'] = df_species_occ_info['Nombre d\'individus'].apply(lambda x: f"{x} individus")

        # Modifier l'ordre des colonnes
        nouvel_ordre = ['Espèce', 'Catégorie', 'Nombre d\'individus', '% dans l\'environnement']
        df_species_occ_info = df_species_occ_info.reindex(columns=nouvel_ordre)

        # Trier les lignes par ordre alphabétique des catégories
        df_trie = df_species_occ_info.sort_values(by='Catégorie')

        # Afficher le DataFrame sur Streamlit sans la colonne des index
        st.markdown(f'''
                <h4>Table d'information de votre environnement</h4>
            ''', unsafe_allow_html=True)
        st.dataframe(df_trie, hide_index=True)

    with col2:
        fig = bar_chart(color_map,0,False)
        # Afficher la figure dans Streamlit
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class=solid>", unsafe_allow_html=True)

    # Generate the report
    report_filename = generate_report(color_map)

    # Download button for the report
    st.download_button("Télécharger le rapport", data=open(report_filename, "rb"), file_name=report_filename)


# -------------------------------------------------------------------------------
# MAIN / HOME PAGE
def main():
    my_logo = add_logo(logo_path="logo.png", width=150, height=150)
    st.sidebar.image(my_logo) 
    # Vérifier si un fichier a été téléchargé
    uploaded_file = st.sidebar.file_uploader("Déposez votre fichier CSV pour analyse, ci-dessous.", type="csv",
                                             accept_multiple_files=False, key=1)
    if uploaded_file:
        df = read_csv(uploaded_file)

        # Vérifier le nombre de colonnes
        if len(df.columns) != 3:
            st.sidebar.error("Le fichier n'est pas correct.")
        else:
            # Renommer les colonnes
            df.columns = ["Species", "Category", "Time"]

            st.sidebar.success("Vous venez de charger un fichier")

            if 'initial_data' not in st.session_state:
                st.session_state.initial_data = df
                st.session_state.view_data = False
            else:
                if compare_dataframes(df, st.session_state.initial_data) == False:
                    st.session_state.initial_data = df
                    st.session_state.view_data = False
                    # st.experimental_rerun()

            # upload_file()
            options = st.sidebar.radio('Choisissez votre page', options=["Page d'accueil", "Individus dans l'environnement",
                                                                        "Pourcentage par catégorie", "Analyse Temporelle",
                                                                        "Rapport général", "Equipe"])

            if options == "Page d'accueil":
                Home_Page()
            # elif options == "Test_home_page_2":
            # Home_Page2()
            # elif options == "Test":
            # test()
            elif options == "Individus dans l'environnement":
                Ind_in_env()
            elif options == "Pourcentage par catégorie":
                Per_by_Category()
            elif options == "Analyse Temporelle":
                Data_and_Time()
            elif options == "Rapport général":
                report()
            elif options == "Equipe":
                Page_team()

            # st.sidebar.markdown("[Linkedin](https://www.linkedin.com/in/th%C3%A9o-masson-b831421a0/)")
            # st.sidebar.markdown("[Spotify](https://www.spotify.com/)")

    else:
        colmargrleft, colmain, colmargeright = st.columns([1, 10, 1], gap="medium")

        with colmain:
            st.markdown(f'''
                <h1 class=page-title>God Saves The Bugs</h1>
                <h3>Bienvenue dans notre application de mesure de la biodiversité !</h3>
                <p class=txt-container-flower>Notre objectif est de vous aider à évaluer et à comprendre la biodiversité d'un lieu spécifique en utilisant des données préalablement collectées et analysées par notre algorithme de machine learning. La biodiversité est essentielle pour maintenir l'équilibre écologique de notre planète, et il est crucial de surveiller et de comprendre les différentes espèces qui peuplent un environnement donné. Grâce à notre algorithme de machine learning, nous avons été en mesure d'extraire des informations précieuses à partir de vastes ensembles de données sur la biodiversité.<br><br>Notre application vous permettra d'explorer ces données et de visualiser les espèces présentes dans le lieu de votre choix. Nous croyons fermement que la technologie et l'intelligence artificielle peuvent jouer un rôle essentiel dans la conservation et la préservation de notre biodiversité. En utilisant les avancées de l'apprentissage automatique, nous sommes en mesure de fournir des données fiables et des visualisations interactives pour vous aider à mieux comprendre et apprécier la richesse biologique d'un lieu. Nous espérons que notre application vous aidera à sensibiliser à l'importance de la biodiversité et à prendre des décisions éclairées pour sa protection.<br><br>Profitez de l'exploration de notre application et découvrez la diversité étonnante de la vie qui peuple notre planète.</p>
                <h5 style='color: #ffffff; backdrop-filter: blur(1px);'>Vous pouvez déposer votre fichier CSV dans la sidebar afin d'accéder aux analyses.</h5>
            ''', unsafe_allow_html=True)



# -------------------------------------------------------------------------------
# MAIN
# Adding a background of the website
# Changer la phot de fond
add_background('background.png')
main()
