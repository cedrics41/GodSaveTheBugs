import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
from datetime import date
import calendar
import time
from PIL import Image, ImageGrab
from zipfile import ZipFile
from base64 import b64encode
from fpdf import FPDF
import tempfile
import plotly.graph_objects as go
import plotly.subplots as sp

# from Master_Project_report_page import report

st.set_page_config(layout="wide")


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
# Create a box with a coloured background
def manage_container(text, back_color, text_align):
    # Peut-être ne pas mettre le texte direct et plutôt faire appel à la fonction markdown_size_center()
    with st.container():
        st.markdown(f'''
        <div style="
            background-color: {back_color};
            padding: 5%;
            border-radius: 6%;
            justify-content: center;
            text-align: {text_align};
            margin: 4%;
            ">
            {text}
        </div>
    ''', unsafe_allow_html=True)


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
    }

    /* Styles CSS pour les conteneurs de texte */
    .txt-container-flower {
        background-color: #808080;
        color: #ffffff;
        border-radius: 25% 10% / 5% 20%;
        padding: 5%;
        margin: 3%;
        text-align: justify;
    }

    /* Styles CSS pour le conteneur général ==> not use */
    .general-container {
        border-radius: 10px;
        margin: 10px;
        padding: 20px;
        display: flex;
        align-items: center;
    }


    /* Styles CSS pour les conteneurs d'indices ==> not use */
    .indice-container {
        background-color: #808080;
        border-radius: 10px;
        padding: 5px;
        margin: 10px;
        border: 3px solid;
        text-align: center;
    }

    /* Style CSS pour box container des différents insectes */
    .Box-container {
        background-color: #808080;
        color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
    }

    /* Style CSS pour les items de la grille de la liste des différents insectes */
    .custom-grid-list-item {
        white-space: nowrap;
        margin: 3%;
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
# Home Page
def Home_Page():
    colmargrleft, colmain, colmargeright = st.columns([1, 10, 1], gap="small")

    with colmain:
        st.markdown(f'''
            <h1 class=page-title>God Saves The Bugs</h1>
            <h3>Bienvenue sur votre espace ! </h3>
            <p class=txt-container-flower>Description à remplir par la suite Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section </p>
        ''', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1], gap="medium")

        with col1:
            # Gérer le taille de la box pour être raccord avec la box de l'indice
            manage_container(
                "Explication de l'indice : \n Ici, nous comptons le nombre d'espèces identifiées (pour nous c'est les insectes différents que nous avons pu identifier + 1 si il y a des espèces non identifiées.Explicatiopn de l'indice: \n Ici, nous comptons le nombre d'espèces identifiés (pour nous c'est les insectes différents que nous avons pu identifier + 1 si il y a des espèces non identifiées.",
                "#808080", "justify")

        with col2:
            # Modifier le texte et les paramètre CSS si besoin + AFFECTER UNE VALEUR A L'INDICE SUITE A UN CALCUL

            species_number = st.session_state.initial_data[st.session_state.initial_data["Species"] != "Non définie"]["Species"].nunique()

            st.markdown(
                """
                <style>
                [data-testid="stMetricLabel"] {
                    font-size: 2em;
                }
                [data-testid="stMetricValue"] {
                    font-size: 6em;
                    color: #008000;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.metric(label="Metric", value=species_number)

        st.divider()

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
def create_pie_chart(column_name):
    # Compter le nombre d'occurrences de chaque nom
    value_counts = st.session_state.initial_data[column_name].value_counts(dropna=False)

    # Créer un DataFrame avec les noms et les occurrences
    data = pd.DataFrame({'Espèces': value_counts.index, 'Occurrences': value_counts.values})

    # Renommer les valeurs nulles en "non-ientifié"
    data['Espèces'].fillna('Non définie', inplace=True)

    # Créer le pie chart interactif avec Plotly Express
    fig = px.pie(data, values='Occurrences', names='Espèces', title='Répartition des espèces observés:')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------------------
# Définir le nombre de colonnes pour affichage 
def nb_col(list):
    item_list = len(list)
    num_col = round(item_list / 4)
    return num_col


# -------------------------------------------------------------------------------
# Créer un bar chart du pourcentage d'espèces en fonction de la catégorie
def bar_chart(color_map):
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
    ))

    # Mettre en forme la figure
    fig.update_layout(
        xaxis_title="Catégorie",
        yaxis_title="Pourcentage",
        showlegend=False,
        title="Titre de la visu",
        title_x=0.5,
        title_font=dict(size=18)
    )

    # Afficher la figure dans Streamlit
    st.plotly_chart(fig, use_container_width=True)


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

        elif selectionFrequency == "Mois":
            filtered_data = df.pivot_table(index='Month', columns=mode, aggfunc='size', fill_value=0)

        elif selectionFrequency == "Journée":
            filtered_data = df.pivot_table(index='Day', columns=mode, aggfunc='size', fill_value=0)

        # Display the line chart
        st.line_chart(filtered_data)


# -------------------------------------------------------------------------------
# Page sur la répartition du nombre d'individus dans l'environnement d'étude
def Ind_in_env():
    colmargrleft, colmain, colmargeright = st.columns([1, 10, 1], gap="medium")

    with colmain:
        list_of_species = get_unique_in_col("Species")

        st.markdown(f'''
            <h1 class=page-title>Individus Répertoriés</h1>
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

        create_pie_chart("Species")

        st.divider()

        st.write("Vous voulez en apprendre davantage sur la répartition des insectes ?")

        description = st.button("En savoir plus...")

        if description:
            st.markdown(f'''
                <p class=txt-container-flower>Équité des espèces : En examinant la répartition des individus entre les espèces, on peut évaluer l'équité des espèces dans l'écosystème. Une répartition équitable indique que les individus sont répartis de manière plus uniforme entre les espèces, tandis qu'une répartition inégale indique une disparité dans l'abondance des espèces.</p>
            ''', unsafe_allow_html=True)


# -------------------------------------------------------------------------------
# Visualisation des ronds
def visu_ronds(df):
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

    # Définir les couleurs pour les catégories
    color_map = {category: f'#{hash(category) % 0xffffff:06x}' for category in categories}

    # Trier le DataFrame par ordre alphabétique des espèces
    nombre_occurrences = nombre_occurrences.sort_values('Species')

    # Effectuer une jointure entre nombre_occurrences et df pour obtenir les catégories correspondantes
    nombre_occurrences = nombre_occurrences.merge(df[['Species', 'Category']].drop_duplicates(), on='Species', how='left')

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
        #width=800,
        #height=250 * row_count,
        autosize = True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=None, # ==> changer la couleur.
        title="Titre de la visu",
        title_x=0.5,
        title_font=dict(size=18)
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
            title='Category',
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

    # Afficher la figure dans Streamlit
    #st.plotly_chart(fig, use_container_width=True)


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

        categories.insert(0, "All")

        filtering_option = st.selectbox("Quelle catégorie voulez-vous observer ?", categories)

        if filtering_option == "All":
            fig, color_map = visu_ronds(st.session_state.initial_data)
            st.plotly_chart(fig, use_container_width=True)
            bar_chart(color_map)
        else:
            filtered_data = st.session_state.initial_data[st.session_state.initial_data['Category'] == filtering_option]
            fig, color_map = visu_ronds(filtered_data)
            st.plotly_chart(fig, use_container_width=True)
        

        st.divider()

        information = st.selectbox(
            "L'importance d'avoir plusieurs catégories d'insectes dans un environnement donné :",
            ('Compréhesion de la biodiversité', "Indicateur de l'état de l'écosystème", 'Conservation'))
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
        "Victoria STASIK - Data engineer",
        "Théo MASSON - Data analyst",
        "Cédric SONG - Data analyst",
        "Mike LEVELEUX - Data engineer",
        "Justine BOILLOT - Software and Hardware specialist",
        "Justine WANG - Hardware specialist",
        "Nouhayla CHERRABI - Data engineer"
    ]

    for col, (team_image, description) in zip(cols, zip(st.session_state.team_images, image_descriptions)):
        col.image(team_image)
        col.write(description)


# -------------------------------------------------------------------------------
# Custom class for the PDF report
class PDF(FPDF):
    def header(self):
        # Add header content here
        self.image("logo.png", x=10, y=10, w=30)
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "My Report", align="C")

    def footer(self):
        # Add footer content here
        pass


# -------------------------------------------------------------------------------
# Function to generate the PDF report ==> A changer en fonction de rapport défini
def generate_report():
    # Create a new PDF document
    pdf = PDF()

    # Set up the document
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add content to the report
    pdf.set_font("Arial", size=12)

    # Generate bar plot
    data = [10, 20, 30, 15, 25]
    categories = ["A", "B", "C", "D", "E"]
    plt.bar(categories, data)
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.title("Bar Plot")
    plt.tight_layout()

    # Create a temporary file for the bar plot image
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        # Save the bar plot as an image in the temporary file
        bar_plot_path = tmpfile.name + ".png"
        plt.savefig(bar_plot_path)
        plt.close()

        # Add the bar plot image to the report
        pdf.cell(0, 10, "Bar Plot", ln=True)
        pdf.image(bar_plot_path, w=150)
        pdf.ln()

    # Add text area
    pdf.multi_cell(0, 10, "This is a text area.")
    pdf.ln()

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

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        #bar_chart()
        st.write("Something is coming....")
    with col2:
        st.write("Something is coming....")

    st.divider()

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
       # bar_chart()
       st.write("Something is coming....")
    with col2:
        st.write("Something is coming....")

    # Generate the report
    report_filename = generate_report()

    # Download button for the report
    st.download_button("Download Report", data=open(report_filename, "rb"), file_name=report_filename)


# -------------------------------------------------------------------------------
# MAIN / HOME PAGE
def main():
    my_logo = add_logo(logo_path="logo.png", width=300, height=90)
    st.sidebar.image(my_logo) 
    # Vérifier si un fichier a été téléchargé
    uploaded_file = st.sidebar.file_uploader("Déposez votre fichier CSV pour analyse, ci-dessous.", type="csv",
                                             accept_multiple_files=False)
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
                <h3>Bienvenue sur notre application intéractive</h3>
                <p class=txt-container-flower>Description à remplir par la suite Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section </p>
                <h5>Vous pouvez déposer votre fichier CSV dans la sidebar afin d'accéder aux analyses.</h5>
            ''', unsafe_allow_html=True)


# -------------------------------------------------------------------------------
# MAIN
main()
