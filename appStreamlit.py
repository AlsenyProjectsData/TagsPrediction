# Core pkges
import streamlit as st
import pandas as pd 
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

##
model_rf = joblib.load(open("TagsPrediction.pkl", "rb"))
## Fct
def predict_tags(docx):
    results = model_rf.predict([docx])
    return results[0]
##
#def get_tags(docx):
    # tags=['.net', 'asp.net','c#','c++','java','javascript','php','python','sql','sql-server']
   #  for col in tags:
      #   print(col)
      #   return tags
## 
def main():
    st.title("Tags Classifier App")
    st.subheader("NLP and ML App whit streamlit")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.markdown("Home-Tags In Text")
        with st.form(key = "tags_clf_from"):
            news_text = st.text_area("Enter Text","Type Here")
            submit_text = st.form_submit_button(label ="Submit")
        if submit_text:
           col1, col2, col3, col4 = st.columns(4)  
## Apply Fct
           prediction = predict_tags(news_text)
          # probability = get_predict_proba(news_text)
           with col1:
                 st.success("ORIGINAL TEXT")
                 st.write(news_text)
                 #st.success("Prediction")
                 #st.write(prediction)
           with col2:  
                 st.success("PREDICTION")   
                 st.write(prediction)
                 
           with col3:
                #st.success("Prediction Probability")
                #st.write(probability)
                 st.success("DETAILS: Le chiffre 1 dans colonne prédiction, correspond à la position des tags prédits par le modèle parmis les 10 tags rangés par ordre dans la colonne TAGS_PREDICTS.")
                 #st.write(prediction)
                 
           with col4:
                st.success("TAGS_PREDICTS : .net, asp.net, c#, c++, java, javascript, php, python, sql, sql-server")
    elif choice == "Monitor":
         st.subheader("Monitor App")
    else:
         st.subheader("About") 
         st.markdown("This apply ")   
          
if __name__ == "__main__":
    main() 
 
