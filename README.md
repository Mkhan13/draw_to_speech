# Draw to Speech AAC Device

## Problem
Nonverbal children often rely on augmentative and alternative communication
(ACC) tools to communicate and express themselves. However, most ACC devices require
navigating menus or recognizing text or icons. This can be diLicult for young children who
communicate more easily through visual symbols or drawings. The goal of this project is to
make an interactive sketch to speech communication board that allows children to draw
simple doodles which the system can classify using deep learning and then generate the
corresponding speech output.

---

## Data Source
- **[The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)**
  - A collection of 50 million drawings across 345 categories
  - 28x28 grayscale images saved as a numpy array grouped by category

---

## Review of Relevant Previous Efforts and Literature  
Some examples of AAC devices are communication boards and interactive tablets
where the user points or clicks on a word to communicate. Communication boards are
rigid because it limits the user to letters, numbers, and a few key phrases. Interactive
tablets allow for more customization and interaction but still often requires the user to
navigate several menus before locating the desired word. No significant research has been done about AI driven AAC devices.

**My Contribution:**  
My project is unique to this field of study and puts a twist on current AAC devices by allowing the user to doodle the word to be said aloud.

---

## Model Evaluation Process & Metric Selection   
- **Metrics:**  
  - Accuracy
  -  Precision
  -  Recall
  -  F1-score
  -  Confusion Matrix
- **Data Splits:** Stratified 80%/10%/10% split for train/validation/test 

All three approaches (naive, classical ML, and deep learning) are trained and evaluated on the same training, validation, and test sets. The results are compared directly against the naive baseline to quantify performance improvements

---

## Modeling Approach  
1. **Naive Baseline:** 
2. **Classical ML Approach:**  
3. **Deep Learning Approach:**  


### Data Processing Pipeline  


The images are then copied into a standardized folder structure under `data/processed/`:  
```
data/processed/
├── train.npz
├── val.npz
├── test.npz
```

### Models Evaluated and Model Selected  
- **Evaluated Models:**


- **Model Selected:**  

### Comparison to Naive Approach  


---

## Visual Interface Demo


Video demo of the project can be found here
Streamlit site can be found here

---

## Results and Conclusions  

---

## Ethics Statement  


---

## Instructions on How to Run the Code

1. Clone the Repository  
`git clone `  
`cd `

3. Install Dependencies  
`pip install -r requirements.txt`

4. Run the Streamlit App  
`streamlit run main.py`  
The app will open in your browser  

6. 


