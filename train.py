from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import json
import keras
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import Input, Embedding, LSTM, Bidirectional, RepeatVector, Dropout, Dense
from keras.models import Model

with open('./dataset1.json', 'r', encoding='utf-8') as file:
    dataset = json.load(file)
    
# Extract pseudocode and C code
pseudocode_samples = [example["Pseudocode"] for example in dataset]
c_code_samples = [example["C_Code"] for example in dataset]

# Tokenize the text data
tokenizer_pseudocode = Tokenizer()
tokenizer_pseudocode.fit_on_texts(pseudocode_samples)


tokenizer_c_code = Tokenizer(filters='')
tokenizer_c_code.fit_on_texts(c_code_samples)

# print(tokenizer_c_code.word_index)

# Convert text to sequences
pseudocode_sequences = tokenizer_pseudocode.texts_to_sequences(pseudocode_samples)
c_code_sequences = tokenizer_c_code.texts_to_sequences(c_code_samples)

# Pad sequences to ensure they have the same length
pseudocode_padded = pad_sequences(pseudocode_sequences)
c_code_padded = pad_sequences(c_code_sequences)

max_length = max(max(len(seq) for seq in pseudocode_sequences), max(len(seq) for seq in c_code_sequences))

pseudocode_sequences_padded = pad_sequences(pseudocode_sequences, maxlen=max_length, padding='post')

c_code_sequences_padded = pad_sequences(c_code_sequences, maxlen=max_length, padding='post')


#----------------------------------------------------------------------------------------------------------------



# Define the model architecture
embedding_dim = 50  # Adjust as needed
latent_dim = 256  # Adjust as needed

# Pseudocode input
pseudocode_inputs = Input(shape=(pseudocode_sequences_padded.shape[1],))
pseudocode_embedding = Embedding(input_dim=len(tokenizer_pseudocode.word_index) + 1, output_dim=embedding_dim)(pseudocode_inputs)


# Add Bidirectional LSTM layer
encoder_lstm_bidirectional = Bidirectional(LSTM(latent_dim,return_sequences=True))(pseudocode_embedding)
encoder_lstm_bidirectional1 = Bidirectional(LSTM(latent_dim))(encoder_lstm_bidirectional)

# encoder_dropout = Dropout(0.2)(encoder_lstm_bidirectional)

# Reshape to repeat the context vector
encoder_output = RepeatVector(pseudocode_sequences_padded.shape[1])(encoder_lstm_bidirectional1)

# Decoder LSTM layer
decoder_lstm = LSTM(latent_dim, return_sequences=True)(encoder_output)
decoder_dropout_1 = Dropout(0.2)(decoder_lstm)

decoder_lstm_2 = LSTM(latent_dim, return_sequences=True)(decoder_dropout_1)
decoder_dropout_2 = Dropout(0.2)(decoder_lstm_2)

decoder_lstm_3 = LSTM(latent_dim, return_sequences=True)(decoder_dropout_2)

# Decoder Bidirectional LSTM layer
decoder_lstm_bidirectional = Bidirectional(LSTM(latent_dim, return_sequences=True))(decoder_lstm_3)



# Dense layer for C code prediction
c_code_prediction = Dense(len(tokenizer_c_code.word_index) + 1, activation='softmax')(decoder_lstm_bidirectional)

# Model
model = Model(inputs=pseudocode_inputs, outputs=c_code_prediction)
opt = keras.optimizers.Adam(learning_rate=0.001)
#Compile the model
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    pseudocode_sequences_padded,
    np.expand_dims(c_code_padded, axis=-1),
    batch_size=8,
    epochs=300,
    validation_split=0.2
)

# # Save the trained model
model.save('lstm_model2.h5')





#------------------------------------------------------------------------------------------------------------------------
################################### TEST      ##################################################

from keras.models import load_model

import numpy as np

# Load the trained model
model = load_model('lstm_model.h5')  # Update with the actual path to your saved model

# Example pseudocode for prediction
test_pseudocode = "Increment the value of x by 1"

# Tokenize and pad the input sequence
input_seq = tokenizer_pseudocode.texts_to_sequences([test_pseudocode])
input_seq_padded = pad_sequences(input_seq, maxlen=pseudocode_sequences_padded.shape[1], padding='post')

# Generate predictions
predicted_probs = model.predict(input_seq_padded)

# Sample indices from the probability distribution
predicted_indices = [np.argmax(seq_probs) for seq_probs in predicted_probs[0]]

# Convert indices back to C code
predicted_c_code = tokenizer_c_code.sequences_to_texts([predicted_indices])[0]

# Print the predicted C code
print(predicted_c_code)

print("-----------------------------------------------------")
# Example pseudocode for prediction
test_pseudocode = "Check if p is true and q is true"

# Tokenize and pad the input sequence
input_seq = tokenizer_pseudocode.texts_to_sequences([test_pseudocode])
input_seq_padded = pad_sequences(input_seq, maxlen=pseudocode_sequences_padded.shape[1], padding='post')

# Generate predictions
predicted_probs = model.predict(input_seq_padded)

# Sample indices from the probability distribution
predicted_indices = [np.argmax(seq_probs) for seq_probs in predicted_probs[0]]

# Convert indices back to C code
predicted_c_code = tokenizer_c_code.sequences_to_texts([predicted_indices])[0]

# Print the predicted C code
print(predicted_c_code)

print("-----------------------------------------------------")
# Example pseudocode for prediction
test_pseudocode = "Shift x left by 3 bits and store the result in x" 

# Tokenize and pad the input sequence
input_seq = tokenizer_pseudocode.texts_to_sequences([test_pseudocode])
input_seq_padded = pad_sequences(input_seq, maxlen=pseudocode_sequences_padded.shape[1], padding='post')

# Generate predictions
predicted_probs = model.predict(input_seq_padded)

# Sample indices from the probability distribution
predicted_indices = [np.argmax(seq_probs) for seq_probs in predicted_probs[0]]

# Convert indices back to C code
predicted_c_code = tokenizer_c_code.sequences_to_texts([predicted_indices])[0]

# Print the predicted C code
print(predicted_c_code)


print("-----------------------------------------------------")
# Example pseudocode for prediction
test_pseudocode = "If a is equal to 0 , execute code block x ; else if a is not equal to 0 , execute code block y"

# Tokenize and pad the input sequence
input_seq = tokenizer_pseudocode.texts_to_sequences([test_pseudocode])
input_seq_padded = pad_sequences(input_seq, maxlen=pseudocode_sequences_padded.shape[1], padding='post')

# Generate predictions
predicted_probs = model.predict(input_seq_padded)

# Sample indices from the probability distribution
predicted_indices = [np.argmax(seq_probs) for seq_probs in predicted_probs[0]]

# Convert indices back to C code
predicted_c_code = tokenizer_c_code.sequences_to_texts([predicted_indices])[0]

# Print the predicted C code
print(predicted_c_code)



print("-----------------------------------------------------")
# Example pseudocode for prediction
test_pseudocode = "Perform bitwise or between m and n and store the result in result"

# Tokenize and pad the input sequence
input_seq = tokenizer_pseudocode.texts_to_sequences([test_pseudocode])
input_seq_padded = pad_sequences(input_seq, maxlen=pseudocode_sequences_padded.shape[1], padding='post')

# Generate predictions
predicted_probs = model.predict(input_seq_padded)

# Sample indices from the probability distribution
predicted_indices = [np.argmax(seq_probs) for seq_probs in predicted_probs[0]]

# Convert indices back to C code
predicted_c_code = tokenizer_c_code.sequences_to_texts([predicted_indices])[0]

# Print the predicted C code
print(predicted_c_code)

