# Core pkges
import streamlit as st
import pandas as pd 
import numpy as np
import sklearn
import joblib

##
model_rf = joblib.load(open("/home/alseny/Documents/ModelFinal/TagsPrediction.pkl", "rb"))
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
                 st.success("DETAILS: La prédiction correspond à une colonne de dix valeurs qui indiquent si un tag a été prédit pour le texte donné (valeur 1) ou non (valeur 0). Chaque valeur correspond à un tag dans l'ordre suivant : .net, asp.net, c#, c++, java, javascript, php, python, sql, sql-server")
                 #st.write(prediction)
                 
    elif choice == "Monitor":
         st.subheader("Monitor App")
    else:
         st.subheader("About") 
         st.markdown("This apply ")   
          
if __name__ == "__main__":
    main() 
 
