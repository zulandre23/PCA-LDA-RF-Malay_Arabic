import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from PIL import Image
import imgaug.augmenters as iaa
from sklearn.pipeline import Pipeline
from itertools import cycle
import time  # Tambahkan modul time

# Patch to replace np.bool with np.bool_
np.bool = np.bool_

# Function to load images and labels
def load_dataset(dataset_dir, image_size=(150, 150)):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path).convert('L').resize(image_size)
            image = np.array(image).flatten()
            images.append(image)
            labels.append(label)
    
    return np.array(images), np.array(labels), class_names

# Function to augment dataset
def augment_dataset(X, y, augment_times=5):
    augmenters = iaa.SomeOf((0, 3), [
        iaa.Fliplr(0.3),    # Flip gambar secara horizontal dengan probabilitas 50%
        iaa.Crop(percent=(0, 0.1)),  # Crop gambar secara acak
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Menambahkan blur Gaussian
        iaa.LinearContrast((0.75, 1.5)),  # Mengubah kontras
        iaa.Multiply((0.8, 1.2)),  # Mengubah kecerahan
        iaa.Affine(rotate=(-20, 20)),  # Rotasi gambar
    ])
    
    X_aug = []
    y_aug = []
    for img, label in zip(X, y):
        img = img.reshape(150, 150) #Bentuk Ulang gambar ke bentuk 2D

        for _ in range(augment_times):
            img_aug= augmenters(image=img)
            X_aug.append(img_aug.flatten())
            y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)

# Load dataset
dataset_dir = "dataset"
X, y, class_names = load_dataset(dataset_dir)

# Augment dataset
X_aug, y_aug = augment_dataset(X, y)
X_combined = np.vstack((X, X_aug))
y_combined = np.hstack((y, y_aug))

# Display dataset size and class distribution
st.write(f"Total dataset size (original): {len(X)}")
class_counts = np.bincount(y)
for i, count in enumerate(class_counts):
    st.write(f"Number of samples in class {class_names[i]} (original): {count}")
class_counts_aug = np.bincount(y_combined)
st.write(f"Total dataset size (augmented): {len(X_combined)}")
for i, count in enumerate(class_counts_aug):
    st.write(f"Number of samples in class {class_names[i]} (augmented): {count}")

# Split dataset into 85% train_temp and 15% test
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X_combined, y_combined, test_size=0.15, random_state=42)

# Split train_temp into 70% train and 15% valid (70% of total data)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_temp, y_train_temp, test_size=0.1765, random_state=42)  # 0.1765 karena 0.15 / 0.85 ~ 0.1765

# Display the sizes of the datasets
class_count_train = np.bincount(y_train)
st.write(f"Total data train: {len(X_train)}")
for i, count in enumerate(class_count_train):
    st.write(f"Number of samples train in class {class_names[i]} (augmented): {count}")

class_count_valid = np.bincount(y_valid)
st.write(f"Total data valid: {len(X_valid)}")
for i, count in enumerate(class_count_valid):
    st.write(f"Number of samples valid in class {class_names[i]} (augmented): {count}")

class_count_test = np.bincount(y_test)
st.write(f"Total data test: {len(X_test)}")
for i, count in enumerate(class_count_test):
    st.write(f"Number of samples test in class {class_names[i]} (augmented): {count}")

# Display original image size and number of features
image_shape = Image.open(os.path.join(dataset_dir, class_names[0], os.listdir(os.path.join(dataset_dir, class_names[0]))[0])).convert('L').resize((150, 150)).size
st.write(f"Ukuran gambar setelah preprocessing: {image_shape}")
st.write(f"Jumlah fitur sebelum PCA: {X_train.shape[1]}")

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
st.write(f"Jumlah fitur setelah normalisasi: {X_train.shape[1]}")

# Function to perform classification with PCA
def classify_with_pca(X_train, X_valid, X_test, y_train, y_valid, y_test):
    start_time = time.time()  # Start timer


    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    X_test_pca = pca.transform(X_test)
    
    st.write(f"Jumlah fitur setelah PCA: {X_train_pca.shape[1]}")

    param_grid = {
        'clf__n_estimators': [100, 200, 300, 350, 400],
        'clf__max_depth': [None, 10, 20, 25, 30],
        'clf__min_samples_split': [2, 5, 10, 15, 20],
        'clf__min_samples_leaf': [1, 2, 4, 5, 6, 8]
    }

    pipeline = Pipeline([
        ('pca', PCA(n_components=0.95)),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2, return_train_score=True)
    grid_search.fit(np.vstack((X_train_pca, X_valid_pca)), np.hstack((y_train, y_valid)))

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    cv_results = grid_search.cv_results_

    classifier = best_estimator
    classifier.fit(np.vstack((X_train_pca, X_valid_pca)), np.hstack((y_train, y_valid)))
    y_pred = classifier.predict(X_test_pca)
    y_score = classifier.predict_proba(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    end_time = time.time()  # End timer
    training_time = end_time - start_time  # Calculate training time

    return accuracy, report, conf_matrix, y_pred, best_params, cv_results, y_score, training_time

# Function to perform classification with LDA
def classify_with_lda(X_train, X_valid, X_test, y_train, y_valid, y_test):
    start_time = time.time()  # Start timer

    lda = LDA()
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_valid_lda = lda.transform(X_valid)
    X_test_lda = lda.transform(X_test)
    
    st.write(f"Jumlah fitur setelah LDA: {X_train_lda.shape[1]}")

    param_grid = {
        'clf__n_estimators': [100, 200, 300, 350, 400],
        'clf__max_depth': [None, 10, 20, 25, 30],
        'clf__min_samples_split': [2, 5, 10, 15, 20],
        'clf__min_samples_leaf': [1, 2, 4, 5, 6, 8]
    }

    pipeline = Pipeline([
        ('lda', LDA()),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2, return_train_score=True)
    grid_search.fit(np.vstack((X_train_lda, X_valid_lda)), np.hstack((y_train, y_valid)))

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    cv_results = grid_search.cv_results_

    classifier = best_estimator
    classifier.fit(np.vstack((X_train_lda, X_valid_lda)), np.hstack((y_train, y_valid)))
    y_pred = classifier.predict(X_test_lda)
    y_score = classifier.predict_proba(X_test_lda)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    end_time = time.time()  # End timer
    training_time = end_time - start_time  # Calculate training time

    return accuracy, report, conf_matrix, y_pred, best_params, cv_results, y_score, training_time

# Function to perform classification with PCA + LDA
def classify_with_pca_lda(X_train, X_valid, X_test, y_train, y_valid, y_test):
    start_time = time.time()  # Start timer


    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    X_test_pca = pca.transform(X_test)
    
    lda = LDA()
    X_train_pca_lda = lda.fit_transform(X_train_pca, y_train)
    X_valid_pca_lda = lda.transform(X_valid_pca)
    X_test_pca_lda = lda.transform(X_test_pca)
    
    st.write(f"Jumlah fitur setelah PCA + LDA: {X_train_pca_lda.shape[1]}")

    param_grid = {
        'clf__n_estimators': [100, 200, 300, 350, 400],
        'clf__max_depth': [None, 10, 20, 25, 30],
        'clf__min_samples_split': [2, 5, 10, 15, 20],
        'clf__min_samples_leaf': [1, 2, 4, 5, 6, 8]
    }

    pipeline = Pipeline([
        ('pca', PCA(n_components=0.95)),
        ('lda', LDA()),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2, return_train_score=True)
    grid_search.fit(np.vstack((X_train_pca_lda, X_valid_pca_lda)), np.hstack((y_train, y_valid)))

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    cv_results = grid_search.cv_results_

    classifier = best_estimator
    classifier.fit(np.vstack((X_train_pca_lda, X_valid_pca_lda)), np.hstack((y_train, y_valid)))
    y_pred = classifier.predict(X_test_pca_lda)
    y_score = classifier.predict_proba(X_test_pca_lda)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    end_time = time.time()  # End timer
    training_time = end_time - start_time  # Calculate training time

    return accuracy, report, conf_matrix, y_pred, best_params, cv_results, y_score, training_time

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Function to plot classification report
def plot_classification_report(report, class_names):
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report[df_report.index.isin(class_names)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    df_report[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax)
    ax.set_xticklabels(df_report.index, rotation=45)
    ax.set_title('Classification Report')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    plt.ylim(0, 1)
    
    st.pyplot(fig)

# Function to plot training history
def plot_training_history(cv_results):
    fig, ax = plt.subplots(2, 1, figsize=(10, 14))

    ax[0].plot(cv_results['mean_train_score'], label='Train Accuracy')
    ax[0].plot(cv_results['mean_test_score'], label='Validation Accuracy')
    ax[0].set_xlabel('Hyperparameter Combination Index')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].legend()

    ax[1].plot(cv_results['mean_fit_time'], label='Training Time')
    ax[1].set_xlabel('Hyperparameter Combination Index')
    ax[1].set_ylabel('Time (seconds)')
    ax[1].set_title('Training Time per Hyperparameter Combination')
    ax[1].legend()

    st.pyplot(fig)

# Function to plot ROC curve
def plot_roc_curve(y_test, y_score, n_classes, class_names):
    # Binarize the output
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.figure(figsize=(10, 7))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Streamlit app
st.title("Klasifikasi Huruf Arab Melayu Menggunakan PCA, LDA, dan Random Forest")

if st.button("Klasifikasi dengan PCA"):
    accuracy, report, conf_matrix, y_pred, best_params, cv_results, y_score, training_time = classify_with_pca(X_train, X_valid, X_test, y_train, y_valid, y_test)
    st.write(f"Akurasi: {accuracy}")
    st.write(f"Parameter terbaik: {best_params}")
    st.write(f"Waktu pelatihan: {training_time:.2f} detik")  # Menampilkan waktu pelatihan
    st.write("Laporan Klasifikasi:")
    st.json(report)
    st.write("Matriks Kebingungan:")
    plot_confusion_matrix(conf_matrix, class_names)
    plot_classification_report(report, class_names)
    st.write("Riwayat Pelatihan:")
    plot_training_history(cv_results)
    st.write("Kurva ROC-AUC:")
    plot_roc_curve(y_test, y_score, len(class_names), class_names)

if st.button("Klasifikasi dengan LDA"):
    accuracy, report, conf_matrix, y_pred, best_params, cv_results, y_score, training_time = classify_with_lda(X_train, X_valid, X_test, y_train, y_valid, y_test)
    st.write(f"Akurasi: {accuracy}")
    st.write(f"Parameter terbaik: {best_params}")
    st.write(f"Waktu pelatihan: {training_time:.2f} detik")  # Menampilkan waktu pelatihan
    st.write("Laporan Klasifikasi:")
    st.json(report)
    st.write("Matriks Kebingungan:")
    plot_confusion_matrix(conf_matrix, class_names)
    plot_classification_report(report, class_names)
    st.write("Riwayat Pelatihan:")
    plot_training_history(cv_results)
    st.write("Kurva ROC-AUC:")
    plot_roc_curve(y_test, y_score, len(class_names), class_names)

if st.button("Klasifikasi dengan PCA + LDA"):
    accuracy, report, conf_matrix, y_pred, best_params, cv_results, y_score, training_time = classify_with_pca_lda(X_train, X_valid, X_test, y_train, y_valid, y_test)
    st.write(f"Akurasi: {accuracy}")
    st.write(f"Parameter terbaik: {best_params}")
    st.write(f"Waktu pelatihan: {training_time:.2f} detik")  # Menampilkan waktu pelatihan
    st.write("Laporan Klasifikasi:")
    st.json(report)
    st.write("Matriks Kebingungan:")
    plot_confusion_matrix(conf_matrix, class_names)
    plot_classification_report(report, class_names)
    st.write("Riwayat Pelatihan:")
    plot_training_history(cv_results)
    st.write("Kurva ROC-AUC:")
    plot_roc_curve(y_test, y_score, len(class_names), class_names)