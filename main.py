import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Прогнозирование мировых продаж видеоигр",
    page_icon="🎮",
    layout="wide"
)

ALL_PLATFORMS = ['Wii', 'NES', 'GB', 'DS', 'X360', 'PS3', 'PS2', 'SNES', 'GBA', '3DS', 'PS4', 'N64', 'PS', 'XB', 'PC', '2600', 'PSP', 'XOne', 'GC', 'WiiU', 'GEN', 'DC', 'PSV', 'SAT', 'SCD', 'WS', 'NG', 'TG16', '3DO', 'GG', 'PCFX']
ALL_GENRES = ['Sports', 'Platform', 'Racing', 'Role-Playing', 'Puzzle', 'Misc', 'Shooter', 'Simulation', 'Action', 'Fighting', 'Adventure', 'Strategy']

TOP_PLATFORMS = ['PS2', 'X360', 'PS3', 'Wii', 'DS']
TOP_GENRES = ['Action', 'Sports', 'Shooter']
TOP_PUBLISHERS = ['Nintendo', 'Electronic Arts', 'Activision Blizzard', 'Sony Computer Entertainment', 'Sony Interactive Entertainment ',
                 'Ubisoft', 'Take-Two Interactive', 'Sega', 'Square Enix', 'Microsoft Game Studios', 'Bethesda Softworks']

@st.cache_resource
def load_model():
    """Загрузка модели"""
    try:
        model = joblib.load('best_xgb.joblib')
        st.success("Модель успешно загружена!")
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

def prepare_features(input_data):
    """Подготовка фичей в том же формате, что и при обучении"""
    try:
        df = pd.DataFrame([input_data])
        
        current_year = datetime.now().year
        df['Age'] = current_year - df['Year']
        df['is_mainstream_platform'] = df['Platform'].isin(TOP_PLATFORMS).astype(int)
        df['is_popular_genre'] = df['Genre'].isin(TOP_GENRES).astype(int)
        
        df['Publisher'] = df['Publisher'].apply(
            lambda x: x if x in TOP_PUBLISHERS else 'Other')
        df['is_AAA_publisher'] = df['Publisher'].isin(
            ['Electronic Arts', 'Nintendo', 'Activision']).astype(int)
        
        for col in ['Platform', 'Genre', 'Publisher']:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
        
        df = df.drop(['Platform', 'Genre', 'Publisher', 'Name'], axis=1, errors='ignore')
        
        return df
    
    except Exception as e:
        st.error(f"Ошибка подготовки данных: {e}")
        return None

def predict_sales(model, input_data):
    """Предсказание продаж"""
    try:
        df_processed = prepare_features(input_data)
        
        if df_processed is None:
            return None
            
        expected_features = model.feature_names_in_
        
        for feature in expected_features:
            if feature not in df_processed.columns:
                df_processed[feature] = 0
        
        df_processed = df_processed[expected_features].astype(float)
        
        log_prediction = model.predict(df_processed)[0]
        
        prediction = np.expm1(log_prediction)
        
        return round(prediction, 2)
    
    except Exception as e:
        st.error(f"Ошибка предсказания: {str(e)}")
        return None

def main():
    st.title("🎮 Прогнозирование мировых продаж видеоигр")
    st.markdown("Введите параметры игры для предсказания мировых продаж")
    
    model = load_model()
    if model is None:
        st.stop()
    
    st.sidebar.header("Параметры игры")
    
    st.sidebar.subheader("📌 Основные характеристики")
    platform = st.sidebar.selectbox("Платформа", options=ALL_PLATFORMS)
    genre = st.sidebar.selectbox("Жанр", options=ALL_GENRES)
    
    publisher = st.sidebar.selectbox(
        "Издатель", 
        options=TOP_PUBLISHERS + ['Другой'],
        index=0
    )
    
    if publisher == 'Другой':
        publisher = st.sidebar.text_input(
            "Введите название издателя (англ.)", 
            max_chars=50,
            help="Введите точное название издателя на английском языке"
        )
        if not publisher:
            st.warning("Пожалуйста, введите название издателя")
            st.stop()
    
    st.sidebar.subheader("📅 Год выпуска")
    year = st.sidebar.slider(
        "Год выпуска", 
        min_value=1980, 
        max_value=datetime.now().year, 
        value=2010
    )
    
    if st.button("Предсказать продажи", type="primary"):
        input_data = {
            'Platform': platform,
            'Genre': genre,
            'Publisher': publisher,
            'Year': year,
            'Name': 'User Input'  
        }
        
        with st.spinner("Вычисляю предсказание..."):
            predicted_sales = predict_sales(model, input_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Введенные параметры")
            params = {
                "Параметр": ["Платформа", "Жанр", "Издатель", "Год выпуска"],
                "Значение": [platform, genre, publisher, str(year)]
            }
            st.table(pd.DataFrame(params))
        
        with col2:
            st.subheader("Результат")
            if predicted_sales is not None:
                st.metric(
                    label="Прогнозируемые мировые продажи",
                    value=f"{predicted_sales} млн копий",
                )
                st.info(f"Это примерно {predicted_sales * 1000000:,.0f} проданных копий")
                
                with st.expander("Детали предсказания"):
                    st.markdown(f"""
                    - **Возраст игры**: {datetime.now().year - year} лет
                    - **Популярная платформа**: {"Да" if platform in TOP_PLATFORMS else "Нет"}
                    - **Популярный жанр**: {"Да" if genre in TOP_GENRES else "Нет"}
                    - **Популярный издатель**: {"Да" if publisher in ['Electronic Arts', 'Nintendo', 'Activision'] else "Нет"}
                    """)
            else:
                st.error("Не удалось выполнить предсказание")

if __name__ == "__main__":
    main()