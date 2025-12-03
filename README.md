# Draw to Speech AAC Device

## Problem


---

## Data Source
- **[The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)**
  - A collection of 50 million drawings across 345 categories

---

## Review of Relevant Previous Efforts and Literature  


**My Contribution:**  


---

## Model Evaluation Process & Metric Selection   
- **Metrics:**  

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


