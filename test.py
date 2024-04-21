import tkinter as tk
from tkinter import ttk
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from keras.preprocessing.text import Tokenizer

# Load the trained model
model = load_model('lstm_model.h5')  # Update with the actual path to your saved model

# Load dataset and preprocess
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

max_length = max(max(len(seq) for seq in tokenizer_pseudocode.texts_to_sequences(pseudocode_samples)),
                 max(len(seq) for seq in tokenizer_c_code.texts_to_sequences(c_code_samples)))

def predict_output(test_pseudocode):
    # Tokenize and pad the input sequence
    input_seq = tokenizer_pseudocode.texts_to_sequences([test_pseudocode])
    input_seq_padded = pad_sequences(input_seq, maxlen=max_length, padding='post')
    
    # Generate predictions
    predicted_probs = model.predict(input_seq_padded)
    
    # Sample indices from the probability distribution
    predicted_indices = [np.argmax(seq_probs) for seq_probs in predicted_probs[0]]
    
    # Convert indices back to C code
    predicted_c_code = tokenizer_c_code.sequences_to_texts([predicted_indices])[0]
    
    return predicted_c_code

def on_predict():
    test_pseudocode = input_text.get("1.0", 'end-1c')  # Get the input text from the Text widget
    predicted_c_code = predict_output(test_pseudocode)
    
    # Display the predicted output
    output_text.delete("1.0", tk.END)  # Clear previous output
    output_text.insert(tk.END, predicted_c_code)

# Create the main window
root = tk.Tk()
root.title("Pseudocode to C Code Converter")
# Input Text Widget
input_label = ttk.Label(root, text="Enter Pseudocode:")
input_label.grid(row=0, column=0, sticky="w")
input_text = tk.Text(root, height=10, width=50)
input_text.grid(row=1, column=0, sticky="nsew")

# Output Text Widget
output_label = ttk.Label(root, text="Predicted C Code:")
output_label.grid(row=0, column=1, sticky="w")
output_text = tk.Text(root, height=10, width=50)
output_text.grid(row=1, column=1, sticky="nsew")

# Button to trigger prediction
predict_button = ttk.Button(root, text="Predict", command=on_predict)
predict_button.grid(row=2, column=0, columnspan=2)

# Make widgets resize with the window
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(1, weight=1)


# Run the main event loop
root.mainloop()
