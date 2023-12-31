# Core pkges
import streamlit as st
import pandas as pd 
import numpy as np
import sklearn
import joblib
##
model_rf = joblib.load(open("/home/alseny/Documents/ModelFinal/TagsPredict.pkl", "rb"))
## Fct
def predict_tags(docx):
    results = model_rf.predict([docx])
    return results[0]
##
#def get_predict_proba(docx):
   # results = model_rf.predict_proba([docx])
   # return results
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
           col1, col2 = st.columns(2)  
## Apply Fct
           prediction = predict_tags(news_text)
          # probability = get_predict_proba(news_text)
           with col1:
                 st.success("Original Text")
                 st.write(news_text)
                 #st.success("Prediction")
                 #st.write(prediction)
           with col2:
                #st.success("Prediction Probability")
                #st.write(probability)
                 st.success("Prediction")
                 st.write(prediction)
    elif choice == "Monitor":
         st.subheader("Monitor App")
    else:
         st.subheader("About") 
         st.markdown("This apply ")   
          
if __name__ == "__main__":
    main() 
 
