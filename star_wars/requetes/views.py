from django.shortcuts import render
from scipy.io import arff
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib, base64
from sklearn.decomposition import PCA
import io
import tensorflow.keras.layers as lyrs
import tensorflow.keras.models as mod
import sklearn.preprocessing as prepro
from sklearn.model_selection import train_test_split

def from_plt_to_img(plot):
    fig = plot.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    return urllib.parse.quote(string)

def home(request):
    return render(request, 'requetes/home.html')

def description(request):
    return render(request, 'requetes/description.html')

def correlation(request):
    data = arff.loadarff('..\\1year.arff')
    df = pd.DataFrame(data[0])
    df.head()
    df = df.dropna()
    features = []
    for i in range(1,65):
        features.append("Attr"+str(i))
    features.append("class")
    x = df.loc[:, features].values
    x = prepro.StandardScaler().fit_transform(x)
    pd.DataFrame(data = x, columns = features).head()
    pca = PCA()
    principalComponents = pca.fit_transform(x)
    principalComponents = pd.DataFrame(principalComponents)
    principalComponents.head()
    explained_variance = pca.explained_variance_ratio_
    compteur = 0
    for i in range(0, 29):
        compteur += explained_variance[i]
    pca = PCA(n_components=29)
    principalComponents = pca.fit_transform(x)
    for i in range(0, 29):
        df['PC' + str(i + 1)] = principalComponents[:, i]
    df.head()
    ind = np.arange(0, 29)
    (fig, ax) = plt.subplots(figsize=(8, 6))
    sns.pointplot(x=ind, y=pca.explained_variance_ratio_)
    ax.set_title('Scree plot')
    ax.set_xticks(ind)
    ax.set_xticklabels(ind)
    ax.set_xlabel('Component Number')
    ax.set_ylabel('Explained Variance')
    context = {}
    context["data"] = from_plt_to_img(plt)
    plt.clf()
    (fig, ax) = plt.subplots(figsize=(8, 8))
    for i in range(0, pca.components_.shape[1]):
        ax.arrow(0,
             0,  # Start the arrow at the origin
             pca.components_[0, i],  #0 for PC1
             pca.components_[1, i],  #1 for PC2
             head_width=0.02,
             head_length=0.02)

        plt.text(pca.components_[0, i] + 0.05,
             pca.components_[1, i] + 0.05,
             df.columns.values[i])
    plt.axis('equal')
    ax.set_title('Variable factor map')
    context["data2"] = from_plt_to_img(plt)
    plt.clf()
    return render(request, 'requetes/correlation.html', context)
    
def classification(request):
    data1 = arff.loadarff('..\\1year.arff')
    df1 = pd.DataFrame(data1[0])
    data2 = arff.loadarff('..\\2year.arff')
    df2 = pd.DataFrame(data2[0])
    data3 = arff.loadarff('..\\3year.arff')
    df3 = pd.DataFrame(data3[0])
    data4 = arff.loadarff('..\\4year.arff')
    df4 = pd.DataFrame(data4[0])
    data5 = arff.loadarff('..\\5year.arff')
    df5 = pd.DataFrame(data5[0])
    frames = [df1, df2, df3, df4, df5]
    df = pd.concat(frames)
    del data1
    del df1
    del data2
    del df2
    del data3
    del df3
    del data4
    del df4
    del data5
    del df5
    df = df.dropna()
    df['class'] = df['class'].astype(int)
    train, test = train_test_split(df, test_size=0.2)
    train_target = train.loc[:, df.columns == 'class']
    test_target = test.loc[:, df.columns == 'class']
    train = train.loc[:, df.columns != 'class']
    test = test.loc[:, df.columns != 'class']

    model = mod.Sequential()
    model.add(lyrs.Dense(500, input_shape = (64,)))
    model.add(lyrs.Activation('relu'))
    model.add(lyrs.Dropout(0.25))
    model.add(lyrs.Dense(250))
    model.add(lyrs.Activation('relu'))
    model.add(lyrs.Dense(1))
    model.add(lyrs.Activation('softmax'))
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train, 
            train_target, 
            epochs=10,
            batch_size = 32, 
            verbose=1, 
            validation_data=(test, test_target))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    context = {}
    context["data"] = from_plt_to_img(plt)
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    context["data2"] = from_plt_to_img(plt)
    plt.clf()
    del model
    model = mod.Sequential()
    model.add(lyrs.Dense(500, input_shape = (64,)))
    model.add(lyrs.Activation('relu'))
    model.add(lyrs.Dropout(0.25))
    model.add(lyrs.Dense(250))
    model.add(lyrs.Activation('relu'))
    model.add(lyrs.Dense(1))
    model.add(lyrs.Activation('sigmoid'))
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train, 
            train_target, 
            epochs=10,
            batch_size = 32, 
            verbose=1, 
            validation_data=(test, test_target))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    context["data3"] = from_plt_to_img(plt)
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    context["data4"] = from_plt_to_img(plt)
    plt.clf()
    return render(request, 'requetes/classification.html', context)