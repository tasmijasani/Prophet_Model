
import streamlit as st
import pandas as pd
import pickle
from prophet import Prophet
from sklearn.preprocessing import StandardScaler

def load_pickled_objects():
   
        model = pickle.load(open('prophet_model.pkl','rb'))
        forecast = pickle.load(open('forecast.pkl','rb'))
        df = pickle.load(open('data.pkl','rb'))
        fig1 = pickle.load(open('fig1.pkl','rb'))

        return model,forecast,df,fig1

def main():
    # Load pickled objects
    model,forecast,df,fig1 = load_pickled_objects()

    # enter data from user
    gen = st.selectbox(
       'Select an option:',
       ('General', 'District')
    )

    if gen == 'General':
      dist = st.selectbox(
       'Select an option:',
       ('No District',)
      )

    
    else:
      dist = st.selectbox(
       'Select an option:',
       ('Jamnagar', 'Banaskantha', 'Rajkot', 'Kachchh', 'Amreli', 'Morbi',
       'Devbhumi-Dwarka', 'Vadodara', 'Sabarkantha', 'Junagadh',
       'Navsari', 'Tapi', 'Anand', 'Chhotaudepur', 'Surat', 'Bhavnagar',
       'Kheda', 'Panchmahals', 'The Dangs', 'Surendranagar', 'Ahmedabad',
       'Patan', 'Gir Somnath', 'Mahisagar', 'Bharuch', 'Valsad', 'Botad',
       'Mahesana', 'Porbandar', 'Aravalli', 'Narmada', 'Gandhinagar',
       'Dahod')
       )
    
    attri = st.selectbox(
       'Select an option:',
       ('No_Of_Victims_Shifted','Cardiac', 'Pregnancy','Respiratory','Trauma','Others')
    )

   
   
    if st.button("submit"):
     
       if gen == "General":
         attri = attri
       else:
         dist = dist
         df = df[df['District_Eng'].str.contains(dist, case=False)]
         attri = attri


       scaler = StandardScaler()
       columns_to_normalize = [attri]
       df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

       dfvar = df[[attri, 'AsOnDt']]
       dfvar = dfvar.rename(columns={'AsOnDt': 'ds', attri: 'y'})

    
       m = Prophet()
       m.fit(dfvar)
       future = m.make_future_dataframe(periods=365)  

       forecast = m.predict(future)
       forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


         # Set up Streamlit app
       st.title('Prophet Forecast Visualization')

       st.subheader('Data')
       st.write(df.head())

         # Display forecast data
       st.subheader('Forecast Data:')
       st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

       fig2 = model.plot_components(forecast)
       st.pyplot(fig2)
   

if __name__ == "__main__":
    main()
