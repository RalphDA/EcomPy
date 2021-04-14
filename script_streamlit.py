


import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV


events = pd.read_csv('events.csv', sep = ',')

st.title('Py-ecommerce')



sommaire = st.sidebar.radio(
    "ETAPES :",
    ('Introduction', '1 - Preprocessing', '2 - Regression linéaire', '3 - Regression régularisée', '4 - Synthèse des résultats'))

#Preprocessing
if sommaire == 'Introduction': 
    
    image5 = Image.open('ecommerce.png')
    st.image(image5, output_format='PNG')
    
    st.header("Modélisation d'un dataset concernant les actions effectuées par des visiteurs sur un site de e-commerce.")

    st.subheader("--> L'objectif est de prédire si les visiteurs vont acheter un produit")
    
    st.write("Après le regroupement des visiteurs en quatre clusters par un algorithme d'apprentissage non supervisé, on s'intéresse maintenant à prédire le nombre de transactions effectuées par un visiteur en fonction du nombre de vues et et du nombre de mises au panier à l'aide d'un modèle de regression linéaire.")
    st.write("On importe les packages et le DataSet 'events'")
    
    image = Image.open('events.png')
    st.image(image, output_format='PNG')
    
elif sommaire =='1 - Preprocessing':
    st.header("Etape 1 Preprocessing : ")

    st.write(" - La variable 'event' est dichotomisée afin de faire ressortir chaque type d'action")
    st.write(" - Les variables d'interêt sont ensuite regroupées dans un nouveau dataset")
    st.write(" - On ne garde que les visiteurs qui ont effectués au moins une transaction sur le site au cours de la période.")
           
    image2 = Image.open('new_events.jpg')
    st.image(image2, output_format='JPG')
    
    st.write(" - On normalise les données avec la méthode Min-Max")
    st.write(" - On stocke la variable 'transaction' de data dans target et le reste des variables dans features. On sépare ensuite les données en un ensemble d'apprentissage et un ensemble de test contenant 20% des données. La reproductibilité de l'aléatoire est fixé à 150.")
    
   

# Entrainement du modèle
elif sommaire == '2 - Regression linéaire':
    transac_df = events.join(pd.get_dummies(events['event'], prefix_sep='_'))
    transac_df_summary = transac_df.groupby('visitorid').agg({'view': 'sum', 'addtocart': 'sum', 'transaction': 'sum'})
    transac_df_summary = transac_df_summary[transac_df_summary['transaction'] >= 1]
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(transac_df_summary), columns= transac_df_summary.columns)
    target = data['transaction']
    features = data.drop('transaction', axis =1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state = 150)
    
    st.header("Etape 2 : Modélisation avec la regression linéaire")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    st.write("Le modèle de regression linéaire estimé s'écrit :")
    st.write("transaction = -0.001258 + 0.221049view + 0.722688addtocart")
    st.write("Le nombre de vues, et plus encore, le nombre de mises au panier augmentent positivement le nombre de transaction effectué par un visiteur sur le site.")
    
    image3 = Image.open('coeff.jpg')
    st.image(image3, output_format='JPG')
    
    st.write("Faisons maintenant le diagnostic du modèle au travers différents tests :ajustement du modèle (coefficient de détermination), analyse des résidus,...)")
    
    st.write("- R2 du modèle-échantillon d'entrainement : ", model.score(X_train, y_train))
    st.write("- R2 obtenu par Cv : ", cross_val_score(model,X_train,y_train).mean())
    st.write("- R2 du modèle-échantillon test : ", model.score(X_test, y_test))
    
    model_pred_train = model.predict(X_train)
    model_pred_test = model.predict(X_test)
    
    st.write("- MSE échantillon-entrainement :", mean_squared_error(model_pred_train, y_train))
    st.write("- MSE échantillon-test :", mean_squared_error(model_pred_test, y_test))
    
    st.write("L'analyse des résidus révèle qu'ils ne sont pas parfaitement disséminés autour de la droite d'équation y=0 que ce soit pour l'échantillon d'entrainement ou de test. Ils sont donc à priori hétérocédastiques. De plus, le diagramme Quantile-Quantile (QQ-Plot) indique clairement que leur distribution ne suit pas une loi normale car ils ne sont pas alignés sur la première bissectrice")
    image4 = Image.open('residu.jpg')
    st.image(image4, output_format='JPG')   
    st.write("Conclusion : le modèle est certes bien ajusté aux données (R2 > 90 %), mais il ne possède pas de bonnes propriétés statistiques (résidus hétérocédastiques et non normaux). De plus, il présente des signes de sur-apprentissage probablement dus à la multicolinéarité entre les variables explicatives.")

elif sommaire == '3 - Regression régularisée':
    st.header("Etape 3 : Régression régularisée")
    st.write("Grâce à la régression régularisée, on peut corriger cet effet de sur-apprentissage dans le modèle. On se propose de tester trois algorithmes (la régression ridge, la régression Lasso et la régression ElasticNet) sur des données qui auront été préalablement standardisées.")
    
    st.write("- On commence par normaliser les variables en utilisant la méthode StandardScaler de la classe sklearn.preprocessing")
    st.write("- On isole la variable cible et les variables explicatives, on scinde les données en deux échantillons. L'échantillon test représente 20% de l'ensemble des observations; la reproductibilité de l'aléatoire est fixé à 280.")
    
    transac_df = events.join(pd.get_dummies(events['event'], prefix_sep='_'))
    transac_df_summary = transac_df.groupby('visitorid').agg({'view': 'sum', 'addtocart': 'sum', 'transaction': 'sum'})
    transac_df_summary = transac_df_summary[transac_df_summary['transaction'] >= 1]
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(transac_df_summary), columns= transac_df_summary.columns)
    target2 = data_scaled['transaction']
    features2 = data_scaled.drop('transaction', axis =1)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(features2, target2, test_size=0.2, random_state = 280)
    
    
    options = ["RidgeCV", "LassoCV", "ElasticNetCV"]
    
    modele_choisi = st.selectbox(label = "Choix de modèle", options = options)
    
    @st.cache
    def train_model(modele_choisi):
        if modele_choisi == options[0]:
            model = RidgeCV(alphas= (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50))
        elif modele_choisi == options[1]:
            model = LassoCV(alphas = [0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50], cv = 10)
        else :
            model = ElasticNetCV(cv=12, l1_ratio = (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50,70), alphas= (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50))
                
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)
        return score
    
    @st.cache
    def train_model2(modele_choisi):
        if modele_choisi == options[0]:
            model = RidgeCV(alphas= (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50))
        elif modele_choisi == options[1]:
            model = LassoCV(alphas = [0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50], cv = 10)
        else :
            model = ElasticNetCV(cv=12, l1_ratio = (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50,70), alphas= (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50))
                
        model.fit(X_train, y_train)
        score2 = model.alpha_
        return score2
    
    @st.cache
    def train_model3(modele_choisi):
        if modele_choisi == options[0]:
            model = RidgeCV(alphas= (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50))
        elif modele_choisi == options[1]:
            model = LassoCV(alphas = [0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50], cv = 10)
        else :
            model = ElasticNetCV(cv=12, l1_ratio = (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50,70), alphas= (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50))
                
        model.fit(X_train, y_train)
        score3 = model.score(X_test, y_test)
        return score3
    
    @st.cache
    def train_model4(modele_choisi):
        if modele_choisi == options[0]:
            model = RidgeCV(alphas= (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50))
        elif modele_choisi == options[1]:
            model = LassoCV(alphas = [0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50], cv = 10)
        else :
            model = ElasticNetCV(cv=12, l1_ratio = (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50,70), alphas= (0.001, 0.05, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 30, 50))
                
        model.fit(X_train, y_train)
        score4 = cross_val_score(model,X_train, y_train).mean()
        return score4
    
    st.write("Score échantillon entrainement : ", train_model(modele_choisi))    
    st.write("Valeur optimale de alpha : ", train_model2(modele_choisi))
    st.write("Score échantillon test : ", train_model3(modele_choisi))
    st.write("Score obtenu par Cv : ", train_model4(modele_choisi))
    
else:
    st.header('Synthèse des résultats')
    
    st.subheader("Voici une vision des résultats par modèle:")
    st.subheader("1. Coefficients des paramètres estimés")
    image6 = Image.open('tab1.jpg')
    st.image(image6, output_format='JPG')
    
    st.subheader("2. Performances des modèles")
    image7 = Image.open('tab2.jpg')
    st.image(image7, output_format='JPG')
    
    st.subheader("3. Erreurs de prédiction")
    image8 = Image.open('tab3.jpg')
    st.image(image8, output_format='JPG')
    
    st.subheader("4. Visualisation des scores")
    image9 = Image.open('tab4.jpg')
    st.image(image9, output_format='JPG')
    
    st.write("Conclusion : La régression Ridge a sensiblement amélioré le modèle par rapport aux modèles. Elle a dans un premier temps renforcé l'influence de la variable 'addtocart' comme facteur explicatif principal des transactions. Elle a augmenté sensiblement la performance du modèle sur l'échantillon test qui est passé de 87.2% à 87.7%. L'erreur de prédiction a chuté considérablement pour les deux échantillons quoique l'écart entre les deux persiste.")
    
    
