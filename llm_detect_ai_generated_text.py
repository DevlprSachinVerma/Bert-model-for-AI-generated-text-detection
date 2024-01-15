import pandas as pd
import numpy as np
import tensorflow as tf

"""# Load Data"""

input_file_path1 = '/content/train_essays.csv'
input_file_path2 = '/content/final.csv'
df1 = pd.read_csv(input_file_path1)
df2 = pd.read_csv(input_file_path2)

print(df1.head(20))
print(len(df1))
print(len(df1['text'][6]))

"""# Combine DataFrames"""

for i in range(len(df2)):
    if df2['label'][i] == 1 and len(df2['text'][i]) > 100:
        df1.loc[len(df1.index)] = [i, 0, df2['text'][i], 1]

test=df1.loc[df1['generated'] == 1]
len(test)

new_df = df1.iloc[0:500,:].copy()
#new_df['generated'].value_counts()

df1=new_df.copy()
#df1['generated'].value_counts()

df1 = df1.append(test)
df1['generated'].value_counts()

print(len(df1))

df1 = df1.drop('prompt_id', axis=1)

df1.info()

df1 = df1.sample(frac=1, random_state=42)
df1.reset_index(drop=True, inplace=True)

"""# Preprocess Text"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
for i in range(len(df1)):
    text = df1['text'][i]
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = ' '.join(filtered_sentence)
    df1['text'][i] = filtered_sentence

"""# Train-Test Split"""

from sklearn.model_selection import train_test_split

X = df1.text
y = df1.generated
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dict1 = {'text': X_train, 'generated':y_train}
dict2 = {'text': X_test, 'generated':y_test}
df_train = pd.DataFrame(dict1)
df_test = pd.DataFrame(dict2)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

"""# Tokenization and Dataset Creation"""

from transformers import TFAutoModel, AutoTokenizer

!pip install datasets
from datasets import Dataset, DatasetDict

model = TFAutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

features = ["text", "generated"]

# Create Datasets for each split
dataset_train = Dataset.from_pandas(df_train[features])
dataset_test = Dataset.from_pandas(df_test[features])

# Create a DatasetDict
dataset_dict = DatasetDict({
    "train": dataset_train,
    "test": dataset_test,
})

# Print the resulting dataset
print(dataset_dict)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

text_encoded = dataset_dict.map(tokenize, batched=True, batch_size=None)

text_encoded.set_format('tf', columns=['input_ids', 'attention_mask', 'token_type_ids', 'generated'])

"""# Model Definition"""

BATCH_SIZE = 10

def order(inp):
    data = list(inp.values())
    return {
        'input_ids': data[1],
        'attention_mask': data[2],
        'token_type_ids': data[3]
    }, data[0]

train_dataset = tf.data.Dataset.from_tensor_slices(text_encoded['train'][:])
train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(text_encoded['test'][:])
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

"""# Model"""

class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)

num_classes = 2
classifier = BERTForClassification(model, num_classes=num_classes)

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

"""# Callbacks"""

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath='/content/bert_model_weights.keras', save_best_only=True)
]

"""# Training"""

try:
    history = classifier.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=3,
        callbacks=callbacks
    )
except Exception as e:
    print(f"Error during training: {str(e)}")

"""# Evaluation and Saving model"""

classifier.evaluate(test_dataset)

classifier.save('/content/bert_model', save_format='tf')

"""# Plot Training history"""

import matplotlib.pyplot as plt
def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

plot_history(history)

y_pred = classifier.predict(test_dataset)

y_pred_labels = np.argmax(y_pred, axis=1)

# True labels
y_true = np.concatenate([y.numpy() for _, y in test_dataset])

from sklearn.metrics import classification_report

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred_labels))

