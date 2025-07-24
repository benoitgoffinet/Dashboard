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


#function
class ViTLayer(Layer):
    def __init__(self, **kwargs):
        super(ViTLayer, self).__init__(**kwargs)
        # ta définition ici

    def call(self, inputs):
        # ta logique ici
        return inputs


class ViTLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vit = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def call(self, inputs):
        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        outputs = self.vit(pixel_values=inputs)
        return outputs.pooler_output

def load_image(filename, base_path, modele):
    filepath = os.path.join(base_path, filename)

    if modele == 'VGG16':
        img = np.load(filepath)
        img = img.astype(np.float32)
        return img

    elif modele == 'VIT':
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))  # adapter selon ta config
        img = np.array(img).astype(np.float32)
        img = (img / 127.5) - 1.0
        return img
    
    else:
        raise ValueError(f"Extension non supportée : {ext}")


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(message)s'
)

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

# Charge modèle

modelV = tf.keras.models.load_model('mon_modele_entraineV.keras')
modelVit = tf.keras.models.load_model(
    'mon_modele_entraineVit.h5',
    custom_objects={'ViTLayer': ViTLayer}
)

# Prédictions et matrice de confusion
y_predV = np.load("ypredv.npy")
cmV = np.load("cmv.npy")


y_predVit = np.load("ypredvit.npy")
cmVit = np.load("cmvit.npy")

# Charge historique
with open("historyV.pkl", "rb") as f:
    historyV = pickle.load(f)

with open("historyVit.pkl", "rb") as f:
    historyVit = pickle.load(f)
    
training_times = {
    "VGG16": 2842.24 ,       # en secondes
    "Vit": 5903    # à adapter selon ton second modèle
}

test_accuracy = {
    "VGG16": 0.94 ,       # en secondes
    "Vit": 0.93   # à adapter selon ton second modèle
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
        title="Matrice de confusion",
        xaxis=dict(title="Prédiction"),
        yaxis=dict(title="Vérité Terrain"),
        annotations = annotations
        
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

def clean_image_path_azure(raw_path):
    # Nettoyage basique du chemin relatif
    if not raw_path.startswith("Images" + os.sep + "Images"):
        cleaned_path = os.path.join("Images", "Images", raw_path)
    else:
        cleaned_path = raw_path

    # Normalise le chemin (convertit les slashes en fonction de l'OS)
    cleaned_path = os.path.normpath(cleaned_path)

    # Remplace les backslashs par des slashes, car les URL doivent utiliser '/'
    cleaned_path = cleaned_path.replace(os.sep, '/')


    return cleaned_path

def encode_image_base64(image_path):
    """
    Ouvre une image puis encode en base64 (sans redimensionnement).
    """
    try:
        
        img = Image.open(image_path)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue()).decode()
        mime = "image/jpeg"
        return f"data:{mime};base64,{encoded}"
    except Exception as e:
        return None

def create_presentation_table(df_all, col_classe='labels', col_img='image_path'):
    total_images = len(df_all)
    total_row = {
        'classe': 'Total',
        'nombre_images': total_images,
        'image_base64': None,
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
        })

    return table_rows





app = dash.Dash(__name__, suppress_callback_exceptions=True)

#  layout doit être défini avant de démarrer l'app
app.layout = html.Div([
    html.H1("Mon Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs(id="tabs", value='tab-presentation', children=[
        dcc.Tab(label='Présentation', value='tab-presentation'),
        dcc.Tab(label='Comparaison', value='tab-comparaison'),
        dcc.Tab(label='VGG16', value='tab-vgg16'),
        dcc.Tab(label='ViT', value='tab-vit')
    ]),
    html.Div(id='tabs-content')
])




@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)

def render_content(tab):
  if tab == 'tab-presentation':
        # Appel de ta fonction qui prépare les données du tableau
        table_rows = create_presentation_table(df_all, col_classe='labels', col_img='image_path')

        # Construction des lignes HTML
        rows = []
        cell_style = {
        'border': '1px solid black',
        'padding': '8px',
        'textAlign': 'center'
        }
        for row in table_rows:
            # Si image_base64 est None (pour la ligne total), on n'affiche pas d'image
            img_cell = html.Td(
                html.Img(
                    src=row['image_base64'],
                    style={'width': '300px', 'height': '300px'},
                    alt=f"Image représentative de la classe {row['classe']}"
                )
            ) if row['image_base64'] else html.Td("")

            rows.append(
                html.Tr([
                    html.Td(row['classe'], style=cell_style),
                    html.Td(row['nombre_images'], style=cell_style),
                    img_cell
                ])
            )
        
        # Création du tableau HTML complet
        table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Classe", style={'textAlign': 'center', 'border': '1px solid black'}),
                    html.Th("Nombre Images", style={'textAlign': 'center', 'border': '1px solid black'}),
                    html.Th("Image", style={'textAlign': 'center', 'border': '1px solid black'})
                ])
            ),
            html.Tbody(rows)
        ], style={
            'border': '1px solid black',
            'borderCollapse': 'collapse',
            'width': '100%',
            'textAlign': 'center'
        })
        return table


  if tab == 'tab-comparaison':
    fig_accuracyV = create_accuracy_graph(historyV, 'VGG16')
    fig_accuracyVit = create_accuracy_graph(historyVit, 'VIT')
    fig_lossV = create_loss_graph(historyV, 'VGG16')
    fig_lossVit = create_loss_graph(historyVit, 'VIT')
    fig_test = create_test_accuracy_bar(test_accuracy)
    fig_time = create_training_time_bar(training_times)

    return html.Div([
        html.Div([
            html.H2("Accuracy Comparaison", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_accuracyV, style={'width': '100%'}),
                ], style={'width': '48%', 'display': 'inline-block'},
                   **{'aria-label': 'Graphique montrant l’Accuracy du modèle VGG16',
                      'aria-describedby': 'desc-graph'}),

                html.Div([
                    dcc.Graph(figure=fig_accuracyVit, style={'width': '100%'}),
                ], style={'width': '48%', 'display': 'inline-block'},
                   **{'aria-label': 'Graphique montrant l’Accuracy du modèle VIT',
                      'aria-describedby': 'desc-graph'}),
            ]),

            html.H2("Loss Comparaison", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_lossV, style={'width': '100%'}),
                ], style={'width': '48%', 'display': 'inline-block'},
                   **{'aria-label': 'Graphique montrant la Loss du modèle VGG16',
                      'aria-describedby': 'desc-loss'}),

                html.Div([
                    dcc.Graph(figure=fig_lossVit, style={'width': '100%'}),
                ], style={'width': '48%', 'display': 'inline-block'},
                   **{'aria-label': 'Graphique montrant la Loss du modèle VIT',
                      'aria-describedby': 'desc-loss'}),
            ]),
            html.Div(
                id='desc-loss',
                children="Ces graphiques comparent la perte (Loss) des modèles VGG16 et VIT au cours de l’entraînement.",
                style={'display': 'none'}
            )
        ], style={'width': '64%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H3("Test Accuracy", style={'textAlign': 'center'}),
            html.Div([
                dcc.Graph(figure=fig_test, style={'width': '100%'})
            ], **{'aria-label': 'Barplot comparant l’accuracy en test des modèles VGG16 et VIT',
                  'aria-describedby': 'desc-graph'}),

            html.H3("Training Time", style={'textAlign': 'center'}),
            html.Div([
                dcc.Graph(figure=fig_time, style={'width': '100%'})
            ], **{'aria-label': 'Barplot comparant le temps d’entraînement des modèles VGG16 et VIT',
                  'aria-describedby': 'desc-graph'})
        ], style={'width': '34%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
    ])

  elif tab == 'tab-vgg16':
    fig_conf, df_report = create_model_result_layout(cmV, y_true, y_predV, class_names, graph_id="confusion-vgg")
    df_report_display = df_report.reset_index().rename(columns={"index": "Classe"})

    return html.Div([
        html.H2("Résultats VGG16", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                dcc.Graph(
                    id="confusion-vgg",
                    figure=fig_conf,
                    config={'displayModeBar': False},
                    clickData=None,
                    style={'width': '100%'}
                )
            ], style={'width': '60%', 'display': 'inline-block'},
               **{
                   'aria-label': 'Matrice de confusion pour le modèle VGG16',
                   'aria-describedby': 'desc-graph'
               }
            ),
            html.Div([
                html.H4("Métriques par classe", style={'textAlign': 'center'}),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in df_report_display.columns],
                    data=df_report_display.to_dict('records'),
                    style_table={'overflowX': 'auto', 'margin': '10px'},
                    style_cell={'textAlign': 'center'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
                )
            ], style={'width': '38%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
        ]),
        html.Div(
            id='desc-graph',
            children="Cette matrice de confusion montre les performances de classification du modèle VGG16 sur l'ensemble de test.",
            style={'display': 'none'}
        )
    ])

  elif tab == 'tab-vit':
    fig_conf, df_report = create_model_result_layout(cmVit, y_true, y_predVit, class_names, graph_id="confusion-vit")
    df_report_display = df_report.reset_index().rename(columns={"index": "Classe"})

    return html.Div([
        html.H2("Résultats VIT", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                dcc.Graph(id="confusion-vit", figure=fig_conf, style={'width': '100%'})
            ], style={'width': '60%', 'display': 'inline-block'},
               **{
                   'aria-label': 'Matrice de confusion pour le modèle VIT',
                   'aria-describedby': 'desc-graph-vit'
               }),
            html.Div([
                html.H4("Métriques par classe", style={'textAlign': 'center'}),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in df_report_display.columns],
                    data=df_report_display.to_dict('records'),
                    style_table={'overflowX': 'auto', 'margin': '10px'},
                    style_cell={'textAlign': 'center'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
                )
            ], style={'width': '38%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
        ]),
        html.Div(
            id='desc-graph-vit',
            children="Cette matrice de confusion montre les performances de classification du modèle VIT sur l'ensemble de test.",
            style={'display': 'none'}
        )
    ])
if __name__ == '__main__':
    app.run(debug=True, port=8051)
  