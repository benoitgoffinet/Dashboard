import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers, Model
from transformers import TFViTModel
from PIL import Image
from sklearn.metrics import classification_report
from dash import dash_table
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import base64
import logging
import sys
import io
from keras.models import load_model


df_train = pd.read_pickle("df_filtré_train.pkl")
df_test = pd.read_pickle("df_filtré_test.pkl")
df_val = pd.read_pickle("df_filtré_val.pkl")
# Concaténer les 3 dataframes
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
counts = df_all['labels'].value_counts().reset_index()
counts.columns = ['Classe', 'Nombre d\'images']
base_pathV = "Images/Images_whitening/"
img_colV = "image_path_whitening"
base_pathVit = "Images/Images_cropping/"
img_colVit = "image_path"
encoded_col = "label_encoded"
label_string_col = "labels"

# Charge labels humains triés par encodage (pour affichage)
label_map = df_test[[encoded_col, label_string_col]].drop_duplicates().sort_values(by=encoded_col)
class_names = label_map[label_string_col].to_list()

# Charge images .npy et normalise


y_true = np.load("ytruev.npy")



# Prédictions et matrice de confusion

y_predV = np.load("ypredv.npy")
cmV = np.load("cmv.npy")


y_predVit = np.load("ypredvit.npy")
cmVit = np.load("cmvit.npy")

def load_history(path):
    with open(path, "rb") as f:
        return pickle.load(f)
# Liste des chemins des fichiers history
history_paths = [
    "historyV_fold1.pkl",
    "historyV_fold2.pkl",
    "historyV_fold3.pkl",
    "historyV_fold4.pkl",
    "historyV_fold5.pkl"
]

history_pathsvit = [
    "historyVit_fold1.pkl",
    "historyVit_fold2.pkl",
    "historyVit_fold3.pkl",
    "historyVit_fold4.pkl",
    "historyVit_fold5.pkl"
]
# Charger tous les historiques
histories = [load_history(p) for p in history_paths]
historiesvit = [load_history(p) for p in history_pathsvit]

# Fusionner les histories en moyennant les métriques epoch par epoch
def average_histories(histories):
    keys = histories[0].keys()
    avg_history = {}
    for key in keys:
        # Liste des listes (une par fold) pour cette métrique
        all_values = [h[key] for h in histories]
        # Calculer la moyenne par epoch
        avg_history[key] = list(np.mean(all_values, axis=0))
    return avg_history

historyV = average_histories(histories)
historyVit = average_histories(historiesvit)

    
training_times = {
    "VGG16": 6609 ,       # en secondes
    "Vit": 15497    
}

test_accuracy = {
    "VGG16": 0.95 ,       # en secondes
    "Vit": 0.99   # à adapter selon ton second modèle
}



# --- Création du dashboard ---

app = dash.Dash(__name__)

def create_confusion_heatmap(cm, classes):
    z = cm.tolist() 
    x = classes
    y = classes

    heatmap = go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Blues',
        showscale=True,
        hovertemplate='Prédit: %{x}<br>Réel: %{y}<br>Valeur: %{z}<extra></extra>'
    )

    # Création des annotations (valeurs numériques dans chaque cellule)
    annotations = []
    for i in range(len(z)):
        for j in range(len(z[i])):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=str(z[i][j]),
                    showarrow=False,
                    font=dict(color='black' if z[i][j] < max(map(max, z))/2 else 'white')
                )
            )

    layout = go.Layout(
        title=dict(
          text="Matrice de confusion",
          x=0.5,           # centre horizontalement (0 = gauche, 1 = droite)
          xanchor='center' # ancre au centre du titre
        ),
        xaxis=dict(title="Prédiction"),
        yaxis=dict(title="Vérité Terrain"),
        annotations=annotations
        
    )

    fig = go.Figure(data=[heatmap], layout=layout)
    return fig

def create_accuracy_graph(history, model):
    epochs = list(range(1, len(history['accuracy']) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history['accuracy'], mode='lines+markers', name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))

    fig.update_layout(
        title = f"Courbes {model}",
        title_x=0.5,
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        legend_title='Métriques'
    )
    return fig

def create_loss_graph(history, model):
    epochs = list(range(1, len(history['loss']) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history['loss'], mode='lines+markers', name='Training Loss'))
    fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], mode='lines+markers', name='Validation Loss'))

    fig.update_layout(
        title = f"Courbes {model}",
        title_x=0.5,
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend_title='Métriques'
    )
    return fig

def create_test_accuracy_bar(test_accuracy_dict):
    fig = go.Figure(
        go.Bar(
            x=list(test_accuracy_dict.keys()),  # noms des modèles
            y=list(test_accuracy_dict.values()),  # leurs accuracies
            marker_color='mediumseagreen'
        )
    )
    fig.update_layout(
        title="Test Accuracy par modèle",
        yaxis=dict(range=[0, 1]),  # axe y de 0 à 1
        yaxis_title="Accuracy",
        xaxis_title="Modèles"
    )
    return fig


    
def create_training_time_bar(training_times):
    models = list(training_times.keys())
    times_min = [t / 60 for t in training_times.values()]  # conversion en minutes

    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=times_min,
            marker_color=["#1f77b4", "#ff7f0e"],
            text=[f"{t:.2f} min" for t in times_min],
            textposition="auto"
        )
    ])
    fig.update_layout(
        title="Temps d'entraînement des modèles",
        xaxis_title="Modèle",
        yaxis_title="Temps (minutes)",
        yaxis=dict(range=[0, max(times_min) * 1.2])
    )
    return fig

def create_model_result_layout(cm, y_true, y_pred, class_names, graph_id="confusion-graph"):
    fig_conf = create_confusion_heatmap(cm, class_names)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    return fig_conf, df_report

def clean_image_path(raw_path):
    

    if not raw_path.startswith("Images" + os.sep + "Images"):
        # Construction propre du chemin
        cleaned_path = os.path.join("Images", "Images", raw_path)
    else:
        cleaned_path = raw_path

    # Normalise (remplace les / ou \ selon l'OS)
    cleaned_path = os.path.normpath(cleaned_path)

    return cleaned_path

def encode_image_base64(image_path):
    """
    Essaie d'ouvrir l'image et de l'encoder en base64.
    En cas d'erreur, renvoie un message d'erreur lisible.
    """
    try:
        img = Image.open(image_path)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue()).decode()
        mime = "image/jpeg"
        return f"data:{mime};base64,{encoded}"
    except Exception as e:
        return f"Erreur: {str(e)}"

def create_presentation_table(df_all, col_classe='labels', col_img='image_path'):
    total_images = len(df_all)
    total_row = {
        'classe': 'Total',
        'nombre_images': total_images,
        'image_base64': None,
        'cleaned_path': None
    }

    grouped = df_all.groupby(col_classe)
    table_rows = [total_row]

    for classe, group in grouped:
        nombre_images = len(group)
        raw_path = group.iloc[0][col_img]
        cleaned_path = clean_image_path(raw_path)

        encoded_img = encode_image_base64(cleaned_path)

        table_rows.append({
            'classe': classe,
            'nombre_images': nombre_images,
            'image_base64': encoded_img,
            'cleaned_path': cleaned_path
        })

    return table_rows
        

theme_style = {
    'fontFamily': 'Arial, system-ui, sans-serif',
    'color': '#003366',             # Texte principal
    'backgroundColor': '#FFFFFF',  # Fond clair pour bon contraste
    'padding': '1.25rem',
    'lineHeight': '1.6'
}
title_style = {
    'fontSize': '1.625rem',
    'fontWeight': 'bold',
    'color': '#003366',
    'marginBottom': '1.25rem',
    'textAlign': 'center',
    'aria-level': '1'
}

subtitle_style = {
    'fontSize': '1.125rem',
    'fontWeight': '600',
    'color': '#003366',
    'marginBottom': '0.625rem',
    'aria-level': '2',
    'textAlign': 'center'
}


menu_label_style = {
    'fontSize': '1rem',  # 16px ≈ 1rem
    'color': '#003366',
    'marginBottom': '0.3125rem',  # 5px ≈ 0.3125rem
    'fontWeight': 'bold'
}

menu_style = {
    'padding': '0.625rem',  # 10px
    'margin': '0.625rem 0',  # 10px vertical
    'border': '2px solid #003366',
    'borderRadius': '5px',
    'backgroundColor': '#FFFFFF',
    'color': '#003366',
    'fontSize': '1rem',
    'textAlign': 'center',
    'width': '100%',
    'outline': '2px solid #003366'
}

cell_style = {
    'border': '1px solid black',
    'padding': '0.5rem',  # 8px ≈ 0.5rem
    'textAlign': 'center',
    'color': '#003366',
    'backgroundColor': '#FFFFFF',
    'fontSize': '0.875rem'  # 14px ≈ 0.875rem
}

header_style = {
    'border': '1px solid black',
    'padding': '0.5rem',
    'textAlign': 'center',
    'fontWeight': 'bold',
    'backgroundColor': '#E6F0FA',
    'color': '#003366'
}

graph_container_style = {
    'width': '48%',
    'display': 'inline-block',
    'padding': '0.625rem',
    'border': '1px solid #bbb',
    'borderRadius': '5px',
    'backgroundColor': '#FAFAFA',
    'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'
}


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Dashboard de classification de races de chiens"

#  layout doit être défini avant de démarrer l'app
app.layout = html.Div([
    html.H1("Dashboard de classification de races de chiens", style=title_style),
    dcc.Tabs(id="tabs", value='tab-presentation', style=menu_style, children=[
        dcc.Tab(label='Présentation', value='tab-presentation', style=menu_label_style),
        dcc.Tab(label='Comparaison', value='tab-comparaison', style=menu_label_style),
        dcc.Tab(label='VGG16', value='tab-vgg16', style=menu_label_style),
        dcc.Tab(label='ViT', value='tab-vit', style=menu_label_style)
    ]),
    html.Div(id='tabs-content')
], style=theme_style)




@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)

def render_content(tab):
   if tab == 'tab-presentation':
    table_rows = create_presentation_table(df_all, col_classe='labels', col_img='image_path')
    print(table_rows)

    rows = []
    for row in table_rows:
        img_cell = html.Img(
            src=row['image_base64'],
            style={
                'maxWidth': '18.75em',
                'maxHeight': '18.75em',
                'width': '100%',
                'height': 'auto',
                'objectFit': 'contain'
            },
            alt=f"Image représentative de la classe {row['classe']}"
        ) if row['image_base64'] else ""

        rows.append(
            html.Tr([
                html.Td(row['classe'], style=cell_style),
                html.Td(row['nombre_images'], style=cell_style),
                html.Td(img_cell, style=cell_style)
            ])
        )

    return html.Div([
     html.H2("Les différentes races de chiens à prédire", style=subtitle_style),
     html.Table([
        html.Thead(
            html.Tr([
                html.Th("Classe", style=header_style),
                html.Th("Nombre Images", style=header_style),
                html.Th("Image", style=header_style)
            ])
        ),
        html.Tbody(rows)
    ], role='table', id='table-presentation', **{
        'aria-label': 'Tableau des différentes races de chiens à prédire',
        'aria-describedby': 'desc-table-presentation'
    }, style={
        'margin': '0 auto',
        'width': '90%',
        'overflowX': 'auto'
    }),
    html.Div(
        "Ce tableau présente les différentes classes, le nombre d’images associées, une image représentative, et le chemin d’accès des fichiers.",
        id='desc-table-presentation',
        style={'display': 'none'}
    )
])


        
      
   if tab == 'tab-comparaison':
    fig_accuracyV = create_accuracy_graph(historyV, 'VGG16')
    fig_accuracyVit = create_accuracy_graph(historyVit, 'VIT')
    fig_lossV = create_loss_graph(historyV, 'VGG16')
    fig_lossVit = create_loss_graph(historyVit, 'VIT')
    fig_test = create_test_accuracy_bar(test_accuracy)
    fig_time = create_training_time_bar(training_times)

    return html.Div([
    html.H1("Comparaison entre le modèle VGG16 et le modèle VIT", style={'textAlign': 'center', 'marginBottom': '2rem'}),
    html.H2("Accuracy Comparaison", style={'textAlign': 'center'}),

    # Première ligne : Accuracy + Test Accuracy
    html.Div([
        html.P(
        "Graphique représentant l’évolution de l’accuracy (précision) à travers les epochs du modèle VGG16 sur le jeu d'entrainement et de validation »",
        id="VGG16-accuracy",
        style={"display": "none"}  # caché visuellement
        ),
        html.Div([
            dcc.Graph(figure=fig_accuracyV, style={'width': '100%'})
        ],
        style=graph_container_style,
        **{
            'aria-label': 'Graphique montrant l’Accuracy du modèle VGG16',
            'aria-describedby': 'VGG16-accuracy'
        }),

        html.Div([
            html.P(
        "Graphique représentant l’évolution de l’accuracy (précision) à travers les epochs du modèle VIT sur le jeu d'entrainement et de validation »",
        id="VIT-accuracy",
        style={"display": "none"}  # caché visuellement
        ),
            dcc.Graph(figure=fig_accuracyVit, style={'width': '100%'})
        ],
        style=graph_container_style,
        **{
            'aria-label': 'Graphique montrant l’Accuracy du modèle VIT',
            'aria-describedby': 'VIT-accuracy'
        }),

        html.Div([
            html.P(
        "Diagramme à barres comparant les résultats d’accuracy sur le jeu de test des modèles VGG16 et ViT »",
        id="test-accuracy",
        style={"display": "none"}  # caché visuellement
        ),
            dcc.Graph(figure=fig_test, style={'width': '100%'})
        ],
        style=graph_container_style,
        **{
            'aria-label': 'Barplot comparant l’accuracy en test des modèles VGG16 et VIT',
            'aria-describedby': 'test-accuracy'
        }),
    ], style={'display': 'flex', 'justifyContent': 'space-around'}),

    html.H2("Loss & Training Time Comparaison", style={'textAlign': 'center'}),

    # Deuxième ligne : Loss + Training Time
    html.Div([
        html.Div([
            html.P(
        "Graphique représentant l’évolution de la loss (précision) à travers les epochs du modèle VGG16 sur le jeu d'entrainement et de validation »",
        id="VGG16-loss",
        style={"display": "none"}  # caché visuellement
        ),
            dcc.Graph(figure=fig_lossV, style={'width': '100%'})
        ],
        style=graph_container_style,
        **{
            'aria-label': 'Graphique montrant la loss du modèle VGG16',
            'aria-describedby': 'VGG16-loss'
        }),

        html.Div([
            html.P(
        "Graphique représentant l’évolution de la loss (précision) à travers les epochs du modèle VIT sur le jeu d'entrainement et de validation »",
        id="VIT-loss",
        style={"display": "none"}  # caché visuellement
        ),
            dcc.Graph(figure=fig_lossVit, style={'width': '100%'})
        ],
        style=graph_container_style,
        **{
            'aria-label': 'Graphique montrant la loss du modèle VIT',
            'aria-describedby': 'VIT-loss'
        }),

        html.Div([
             html.P(
        "Diagramme en barres comparant les temps d’entraînement des modèles VGG16 et ViT»",
        id="time-train",
        style={"display": "none"}  # caché visuellement
        ),
            dcc.Graph(figure=fig_time, style={'width': '100%'})
        ],
        style=graph_container_style,
        **{
            'aria-label': 'Barplot comparant le temps d’entraînement des modèles VGG16 et VIT',
            'aria-describedby': 'time-train'
        }),
    ], style={'display': 'flex', 'justifyContent': 'space-around'}),

    html.Div(
        id='desc-loss',
        children="Ces graphiques comparent la perte (Loss) pendant l'entraînement et le temps total d’entraînement pour les modèles VGG16 et VIT.",
        style={'position': 'absolute', 'left': '-9999px'}
    )
])

  
    
   elif tab == 'tab-vgg16':
    fig_conf, df_report = create_model_result_layout(cmV, y_true, y_predV, class_names, graph_id="confusion-vgg")
    df_report_display = df_report.reset_index().rename(columns={"index": "Classe"})

    return html.Div([
    html.H2("Résultats VGG16", style=subtitle_style),
    html.Div([
        dcc.Graph(
            id="confusion-vgg",
            figure=fig_conf,
            config={'displayModeBar': False},
            clickData=None
        )
    ], style={
        **graph_container_style,
        'width': 'max',
        'margin': '0 auto',
        'textAlign': 'center',
        'display': 'block' 
    },
       **{
           'aria-label': 'Matrice de confusion pour le modèle VGG16',
           'aria-describedby': 'VGG16-confusion'
       }
    ),
    html.Div(
        id='VGG16-confusion',
        children="Cette matrice de confusion montre les performances de classification du modèle VGG16 sur l'ensemble de test.",
        style={'display': 'none'}
    )
])

   elif tab == 'tab-vit':
    fig_conf, df_report = create_model_result_layout(cmVit, y_true, y_predVit, class_names, graph_id="confusion-vit")
    df_report_display = df_report.reset_index().rename(columns={"index": "Classe"})

    return html.Div([
        html.H2("Résultats VIT", style= subtitle_style),
        html.Div([
            html.Div([
                dcc.Graph(
                    id="confusion-vit",
                    figure=fig_conf,
                    config={'displayModeBar': False},
                    clickData=None
                )
            ], style=
                     {**graph_container_style,
                     'width': 'max',
                     'margin': '0 auto',
                     'textAlign': 'center',
                     'display': 'block' 

               },
               **{
                   'aria-label': 'Matrice de confusion pour le modèle VIT',
                   'aria-describedby': 'VIT-confusion'
               }),
        ]),
        html.Div(
            id='VIT-confusion',
            children="Cette matrice de confusion montre les performances de classification du modèle VIT sur l'ensemble de test.",
            style={'display': 'none'}
        )
    ])




if __name__ == '__main__':
    app.run(debug=True, port=8051)
 