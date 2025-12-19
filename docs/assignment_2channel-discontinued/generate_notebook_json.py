import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phase 4: Subvocalization Detection - MaxCRNN Training Pipeline\n",
    "\n",
    "**Hardware:** A100 GPU (Recommended)\n",
    "**Goal:** Train on Level 3 (Mouthing) -> Test on Level 4 (Silent Articulation)\n",
    "\n",
    "## 1. Setup\n",
    "Make sure you have uploaded `Phase4_Data.zip` to your Google Drive root."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mount Google Drive\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# Unzip Data\n",
    "!unzip -q /content/drive/MyDrive/Phase4_Data.zip -d /content/data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import glob\n",
    "import os\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras import layers, Model\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "print(f\"TensorFlow Version: {tf.__version__}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Loading & Preprocessing\n",
    "We load data from the `Level_3_Mouthing` (Train) and `Level_4_Silent` (Test) folders."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Constants\n",
    "SEQ_LEN = 1000  # 1 second @ 1000Hz\n",
    "CHANNELS = 2\n",
    "CLASSES = ['GHOST', 'LEFT', 'STOP', 'REST']\n",
    "CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}\n",
    "\n",
    "def load_level_data(base_path, level_name):\n",
    "    X = []\n",
    "    y = []\n",
    "    \n",
    "    # Recursive search for CSVs in the level folder\n",
    "    pattern = os.path.join(base_path, \"**\", level_name, \"*.csv\")\n",
    "    files = glob.glob(pattern, recursive=True)\n",
    "    \n",
    "    print(f\"Found {len(files)} files for {level_name}\")\n",
    "    \n",
    "    for f in files:\n",
    "        # Extract label from filename (e.g., \"GHOST_01.csv\")\n",
    "        filename = os.path.basename(f)\n",
    "        label_str = filename.split('_')[0]\n",
    "        \n",
    "        if label_str not in CLASS_MAP:\n",
    "            continue\n",
    "            \n",
    "        df = pd.read_csv(f, header=None)\n",
    "        \n",
    "        # Ensure 1000 samples\n",
    "        data = df.values[:SEQ_LEN]\n",
    "        if len(data) < SEQ_LEN:\n",
    "            # Pad if short\n",
    "            pad = np.zeros((SEQ_LEN - len(data), CHANNELS))\n",
    "            data = np.vstack([data, pad])\n",
    "            \n",
    "        X.append(data)\n",
    "        y.append(CLASS_MAP[label_str])\n",
    "        \n",
    "    return np.array(X), np.array(y)\n",
    "\n",
    "# Load Train (Open Mouth) and Test (Closed Mouth)\n",
    "data_root = \"/content/data\"\n",
    "X_train, y_train = load_level_data(data_root, \"Level_3_Mouthing\")\n",
    "X_test, y_test = load_level_data(data_root, \"Level_4_Silent\")\n",
    "\n",
    "print(f\"Train Shape: {X_train.shape}\")\n",
    "print(f\"Test Shape: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. MaxCRNN Architecture\n",
    "Implementing the Inception + Bi-LSTM + Attention architecture."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def inception_block(x, filters):\n",
    "    conv1 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)\n",
    "    conv3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)\n",
    "    conv5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(x)\n",
    "    pool = layers.MaxPooling1D(3, strides=1, padding='same')(x)\n",
    "    pool = layers.Conv1D(filters, 1, padding='same', activation='relu')(pool)\n",
    "    return layers.Concatenate()([conv1, conv3, conv5, pool])\n",
    "\n",
    "def build_maxcrnn(input_shape=(1000, 2), n_classes=4):\n",
    "    inputs = layers.Input(shape=input_shape)\n",
    "    \n",
    "    # Feature Extraction\n",
    "    x = inception_block(inputs, 64)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Dropout(0.3)(x)\n",
    "    \n",
    "    x = inception_block(x, 128)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Dropout(0.3)(x)\n",
    "    \n",
    "    # Temporal Modeling\n",
    "    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)\n",
    "    \n",
    "    # Attention\n",
    "    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)\n",
    "    \n",
    "    # Classifier\n",
    "    x = layers.GlobalAveragePooling1D()(x)\n",
    "    x = layers.Dense(64, activation='relu')(x)\n",
    "    x = layers.Dropout(0.5)(x)\n",
    "    outputs = layers.Dense(n_classes, activation='softmax')(x)\n",
    "    \n",
    "    return Model(inputs, outputs, name='MaxCRNN')\n",
    "\n",
    "model = build_maxcrnn()\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compile\n",
    "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# callbacks\n",
    "early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)\n",
    "\n",
    "# Train on Level 3 (Mouthing)\n",
    "history = model.fit(\n",
    "    X_train, y_train,\n",
    "    validation_split=0.2,\n",
    "    epochs=100,\n",
    "    batch_size=32,\n",
    "    callbacks=[early_stop]\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Evaluation on Target Domain (Silence)\n",
    "Testing strictly on Level 4 data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate\n",
    "loss, acc = model.evaluate(X_test, y_test)\n",
    "print(f\"\\nTest Accuracy (Silent Articulation): {acc*100:.2f}%\")\n",
    "\n",
    "# Confusion Matrix\n",
    "y_pred_prob = model.predict(X_test)\n",
    "y_pred = np.argmax(y_pred_prob, axis=1)\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(8,6))\n",
    "sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')\n",
    "plt.title('Confusion Matrix: Silent Articulation Test')\n",
    "plt.ylabel('True (Level 4)')\n",
    "plt.xlabel('Predicted')\n",
    "plt.show()\n",
    "\n",
    "print(classification_report(y_test, y_pred, target_names=CLASSES))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Random Forest Benchmark (Deployed Model)\n",
    "Checking if simple stats features work on the Manifold."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "def extract_features(X):\n",
    "    # Simple stats: RMS, Std, Max\n",
    "    # X shape: (N, 1000, 2)\n",
    "    features = []\n",
    "    for sample in X:\n",
    "        f = []\n",
    "        for ch in range(2):\n",
    "            s = sample[:, ch]\n",
    "            f.extend([np.sqrt(np.mean(s**2)), np.std(s), np.max(s)])\n",
    "        features.append(f)\n",
    "    return np.array(features)\n",
    "\n",
    "X_train_feats = extract_features(X_train)\n",
    "X_test_feats = extract_features(X_test)\n",
    "\n",
    "rf = RandomForestClassifier(n_estimators=100)\n",
    "rf.fit(X_train_feats, y_train)\n",
    "\n",
    "print(f\"RF Test Accuracy: {rf.score(X_test_feats, y_test)*100:.2f}%\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('/Users/cvk/Downloads/carl/phase1-5/CP-PHASE4_Subvocalization_25TPE/docs/assignment/MaxCRNN_Colab_Training.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook generated successfully!")
