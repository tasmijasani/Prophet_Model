import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from prophet import Prophet
from sklearn.preprocessing import StandardScaler

def load_pickled_objects():
   
        model = pickle.load(open('prophet_model.pkl','rb'))
        df = pickle.load(open('data.pkl','rb'))
        deepar_model = pickle.load(open('deepAr_model.pkl','rb'))

        return model,df,deepar_model

def filter_by_d(df, selected,attribute):
    filtered_df = df[df['District_Eng'] == selected]
    filtered_df = filtered_df[['Taluka_Eng', attribute]]
    filtered_df = filtered_df.groupby('Taluka_Eng').agg({attribute: 'sum'}).reset_index()
    return(filtered_df)

#style
def centered_title(title):
    return f"<h1 style='text-align: center;margin-top:10%;margin-bottom:4%'>{title}</h1>"



def main():
    # Load pickled objects
    model,df,deepar_model = load_pickled_objects()
    df1= df
    df['AsOnDt'] = pd.to_datetime(df['AsOnDt'])
    df['AsOnDt'] = df['AsOnDt'].dt.strftime('%Y-%m-%d %H:%M')
    
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
      place='District_Eng'

      
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
      place=dist
      
    
    attri = st.selectbox(
       'Select an option:',
       ('No_Of_Victims_Shifted','Cardiac', 'Pregnancy','Respiratory','Trauma','Others')
    )

  
    col1, col2,col3 = st.columns(3)
    with col1:
      button1 = st.button("Prophet")
    with col2:
      button2 = st.button("Deep AR")  
    with col3:
      button3 = st.button("Visulization")
   
    if button1:
     
       if gen == "General":
         attri = attri
         df=df.groupby('AsOnDt').agg({attri: 'sum'}).reset_index()
         dist = 'District_Eng'
       else:
         dist = dist
         df = df[df['District_Eng'].str.contains(dist, case=False)]
         df = df.groupby(['District_Eng', 'AsOnDt']).agg({attri: 'sum'}).reset_index()
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
       title = "Prophet Forecast Visualization"
       st.markdown(centered_title(title), unsafe_allow_html=True)
       st.subheader('Data')
       st.write(df.head())

       # Display forecast data
       st.subheader('Forecast Data:')
       st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

       fig2 = model.plot_components(forecast)
       st.pyplot(fig2)
       fig1=m.plot(forecast)
       st.pyplot(fig1)   
       
    if button2:
       df['AsOnDt'] = pd.to_datetime(df['AsOnDt'])
       df['AsOnDt'] = df['AsOnDt'].dt.strftime('%Y-%m')

       if gen == "General":
         attri = attri
         df=df.groupby('AsOnDt').agg({attri: 'sum'}).reset_index()
       else:
         dist = dist
         attri = attri
         df = df[df['District_Eng'].str.contains(dist, case=False)]
         df = df.groupby(['District_Eng', 'AsOnDt']).agg({attri: 'sum'}).reset_index()
         
     
       dfvar = df[[attri,'AsOnDt']] 
       dfvar = dfvar.rename(columns={'AsOnDt': 'ds', attri: 'y'})
       dfvar['ds'] = pd.to_datetime(dfvar['ds'])
       dfvar.set_index('ds', inplace=True)

       dataset = PandasDataset(dfvar, freq="1M", target="y")
       tot = 30 
       part = 10 
       
       training_data, test_gen = split(dataset, offset=-tot)
       test_data = test_gen.generate_instances(prediction_length=part, windows=3)
      
       title = "DeepAr Forecast Visualization"
       st.markdown(centered_title(title), unsafe_allow_html=True)
       forecasts = list(deepar_model.predict(test_data.input))
       fig, ax = plt.subplots(figsize=(12, 8))
       for forecast in forecasts:
          fig1= forecast.plot(ax=ax)
       st.pyplot(fig1)

    if button3:
        if gen == "General":
          attri = attri
          place = 'District_Eng'

        else:
          df1 = filter_by_d(df1, dist,attri)
          place = 'Taluka_Eng'
          attri = attri
          
        title = "Data Visualization"
        st.markdown(centered_title(title), unsafe_allow_html=True)          
        # Bar plot
        st.subheader('Bar Plot')
        st.bar_chart(x=place, y=attri, data=df1)

        # Pie Chart
        st.subheader('Pie Chart')
        fig, ax = plt.subplots()
        ax.pie(df1.groupby(place)[attri].sum(), labels=df1[place].unique(), autopct='%1.1f%%')
        st.pyplot(fig)

        # Histrogram
        st.subheader('Histogram')
        fig, ax = plt.subplots()
        ax.hist(df1[attri], bins=20, edgecolor='black')
        plt.xticks(rotation=90)
        st.pyplot(fig)
       

if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()


