from ML import train, filter1, prediction
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.image import Image
from kivy.graphics import Color, Rectangle
import requests
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
import pandas as pd
import numpy as np
from kivy.uix.gridlayout import GridLayout
import cv2
from keras.preprocessing import image
from kivy.uix.camera import Camera
import kivy
from kivy.app import App
from kivy.uix.image import Image
import tensorflow as tf
df_full = pd.read_csv("Animal Dataset.csv") 
#train() 
kivy.require('1.11.1')

# Cargar el modelo preentrenado (puedes usar un modelo como MobileNet o cualquier otro modelo de clasificación de animales)
# Fuera de cualquier clase
model = tf.keras.applications.MobileNetV2(weights='imagenet')


diet_map = {diet: code for code, diet in enumerate(df_full['Diet'].astype('category').cat.categories)}

# API key de NewsAPI (reemplázala por la tuya)
API_KEY = 'ebc4df9d18bb4097a3d79b470fd1fef6'
URL = 'https://newsapi.org/v2/everything'

# Función para obtener noticias sobre animales
def get_animal_news(query='animals'):
    params = {
        'q': query,  # Buscar noticias sobre animales
        'apiKey': API_KEY,
        'language': 'es',  # Puedes cambiar el idioma aquí
    }
    response = requests.get(URL, params=params)
    data = response.json()
    return data['articles']  # Devuelve las noticias
class AnimalClassifierScreen(Screen):
    def __init__(self, **kwargs):
        super(AnimalClassifierScreen, self).__init__(**kwargs)
        self.model = tf.keras.applications.MobileNetV2(weights='imagenet')
        self.build_ui()

    def build_ui(self):

        layout = BoxLayout(orientation='vertical')

        # Layout horizontal para la cámara y el botón de captura
        camera_layout = BoxLayout(orientation='vertical')

        # Crear el componente de la cámara
        self.camera = Camera(play=True)
        camera_layout.add_widget(self.camera)

        # Crear un botón para capturar la imagen
        self.capture_button = Button(text='Capture Image')
        self.capture_button.bind(on_press=self.capture_image)
        camera_layout.add_widget(self.capture_button)

        layout.add_widget(camera_layout)

        # Crear una etiqueta para mostrar el resultado
        self.result_label = Label(text='Prediction: None')
        layout.add_widget(self.result_label)

        # Botón para volver atrás
        back_button = Button(text="Back", size_hint=(None, None), width=100, height=50)
        back_button.bind(on_release=lambda x: setattr(self.manager, 'current', 'main_screen'))
        layout.add_widget(back_button)

        self.add_widget(layout)

    def capture_image(self, instance):
        # Capturar la imagen de la cámara
        texture = self.camera.texture
        image_data = texture.pixels
        
        # Convertir los datos de la cámara a una imagen
        width, height = texture.size
        img_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))

        # Convertir la imagen a RGB (si es necesario)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Preprocesar la imagen para el modelo de IA
        img = cv2.resize(img_array, (224, 224))  # Redimensionar para MobileNetV2
        img = np.expand_dims(img, axis=0)  # Agregar la dimensión del batch
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # Preprocesar la imagen

        # Realizar la predicción
        predictions = self.model.predict(img)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)

        # Mostrar el resultado de la predicción
        best_prediction = decoded_predictions[0][0]
        animal_name = best_prediction[1]
        confidence = best_prediction[2]

        self.result_label.text = f'Prediction: {animal_name}\nConfidence: {confidence*100:.2f}%'

class AddAnimal(Screen):
    def __init__(self, **kwargs):
        super(AddAnimal, self).__init__(**kwargs)
        self.build_ui()

    def build_ui(self):
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)  # Cambiar layout a self.layout

        with self.layout.canvas.before:
            Color(0.82, 0.71, 0.55, 1)  # Marrón clarito
            self.rect = Rectangle(size=self.layout.size, pos=self.layout.pos)
            self.layout.bind(size=self._update_rect, pos=self._update_rect)

        # Crear los TextInput para cada parámetro
        self.input1 = TextInput(hint_text='Animal', size_hint=(None, None), width=300, height=50)
        self.input2 = TextInput(hint_text='Height (cm)', size_hint=(None, None), width=300, height=50)
        self.input3 = TextInput(hint_text='Weight (kg)', size_hint=(None, None), width=300, height=50)
        self.input4 = TextInput(hint_text='Color', size_hint=(None, None), width=300, height=50)
        self.input5 = TextInput(hint_text='Lifespan (years)', size_hint=(None, None), width=300, height=50)
        self.input6 = TextInput(hint_text='Diet', size_hint=(None, None), width=300, height=50)
        self.input7 = TextInput(hint_text='Habitat', size_hint=(None, None), width=300, height=50)
        self.input8 = TextInput(hint_text='Predators', size_hint=(None, None), width=300, height=50)
        self.input9 = TextInput(hint_text='Average Speed (km/h)', size_hint=(None, None), width=300, height=50)
        self.input10 = TextInput(hint_text='Countries Found', size_hint=(None, None), width=300, height=50)
        self.input11 = TextInput(hint_text='Conservation Status', size_hint=(None, None), width=300, height=50)
        self.input12 = TextInput(hint_text='Top Speed (km/h)', size_hint=(None, None), width=300, height=50)
        self.input13 = TextInput(hint_text='Endangered y/n', size_hint=(None, None), width=300, height=50)

        # Agregar los TextInput al layout
        self.layout.add_widget(self.input1)
        self.layout.add_widget(self.input2)
        self.layout.add_widget(self.input3)
        self.layout.add_widget(self.input4)
        self.layout.add_widget(self.input5)
        self.layout.add_widget(self.input6)
        self.layout.add_widget(self.input7)
        self.layout.add_widget(self.input8)
        self.layout.add_widget(self.input9)
        self.layout.add_widget(self.input10)
        self.layout.add_widget(self.input11)
        self.layout.add_widget(self.input12)
        self.layout.add_widget(self.input13)

        # Botón para agregar el animal
        add_button = Button(text='Add Specie', size_hint=(None, None), width=200, height=50)
        add_button.bind(on_release=self.save_values)
        self.layout.add_widget(add_button)

        # Botón para volver atrás
        back_button = Button(text="Back", size_hint=(None, None), width=100, height=50)
        back_button.bind(on_release=lambda x: setattr(self.manager, 'current', 'main_screen'))
        self.layout.add_widget(back_button)

        self.add_widget(self.layout)

    def save_values(self, instance):
        a = self.input1.text
        b = self.input2.text
        c = self.input3.text
        d = self.input4.text
        e = self.input5.text
        f = self.input6.text
        g = self.input7.text
        h = self.input8.text
        i = self.input9.text
        j = self.input10.text
        k = self.input11.text
        l = self.input12.text
        m = self.input13.text

        if m.lower() == 'y':
            m = 1
        elif m.lower() == 'n':
            m = 0

        new_row = {
            'Animal': a,
            'Height (cm)': b,
            'Weight (kg)': c,
            'Color': d,
            'Lifespan (years)': e,
            'Diet': f,
            'Habitat': g,
            'Predators': h,
            'Average Speed (km/h)': i,
            'Countries Found': j,
            'Conservation Status': k,
            'Top Speed (km/h)': l,
            'Endangered': m
        }

        global df_full
        df_full = pd.concat([df_full, pd.DataFrame([new_row])], ignore_index=True)
        
        # Crear el mensaje de éxito
        self.result_label = Label(text='Animal added successfully!', size_hint_y=None, height=50)
        df_full = pd.read_csv("Animal Dataset.csv") 
        self.layout.add_widget(self.result_label)  # Aquí ahora puedes usar self.layout

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos
class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.build_ui()

    def build_ui(self):
        # Crear un layout principal
        layout = BoxLayout(orientation='vertical')

        with layout.canvas.before:
            Color(0.75, 0.84, 0.18, 1)  # Establecer el color de fondo en verde
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
            layout.bind(size=self._update_rect, pos=self._update_rect)

        # Añadir un logo
        anchor_layout = AnchorLayout(anchor_x='center', anchor_y='top', size_hint=(1, None), height=200)
        logo = Image(source='Wild Buddy Logo.png', size_hint=(None, None), size=(300, 200))
        anchor_layout.add_widget(logo)
        layout.add_widget(anchor_layout)

        # Crear un botón para abrir el menú desplegable
        menu_button = Button(text="Menu", size_hint=(None, None), width=300, height=50)

        # Crear el menú desplegable
        dropdown = DropDown()

                # Opciones del menú
        option1 = Button(text='WildLife Search', size_hint_y=None, height=44)
        option1.bind(on_release=lambda btn: setattr(self.manager, 'current', 'search_screen'))  # Este nombre debe coincidir
        dropdown.add_widget(option1)

        option2 = Button(text='WildLife Upload', size_hint_y=None, height=44)
        option2.bind(on_release=lambda btn: setattr(self.manager, 'current', 'orange_screen'))  # 'orange_screen' debe coincidir
        dropdown.add_widget(option2)

        option3 = Button(text='Species Recognition', size_hint_y=None, height=44)
        option3.bind(on_release=lambda btn: setattr(self.manager, 'current', 'red_screen'))  # 'red_screen' debe coincidir
        dropdown.add_widget(option3)

        option4 = Button(text='Extinction Checker', size_hint_y=None, height=44)
        option4.bind(on_release=lambda btn: setattr(self.manager, 'current', 'green_screen'))  # 'green_screen' debe coincidir
        dropdown.add_widget(option4)



        # Abrir el menú desplegable cuando se haga clic en el botón
        menu_button.bind(on_release=dropdown.open)

        # Crear un área para el contenido principal
        content = BoxLayout(orientation='vertical', size_hint_y=None, height=500)
        content.bind(minimum_height=content.setter('height'))

        # Obtener las noticias
        articles = get_animal_news()

        # Añadir las noticias al layout (sin las imágenes)
        for article in articles:
            news_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=150)
            
            # Título y descripción
            title_label = Label(text=article['title'], size_hint_y=None, height=50, bold=True)
            description_label = Label(text=article['description'], size_hint_y=None, height=50)

            news_layout.add_widget(title_label)
            news_layout.add_widget(description_label)

            content.add_widget(news_layout)

        scroll_view = ScrollView()
        scroll_view.add_widget(content)
        
        # Añadir el botón del menú y el contenido al layout principal
        layout.add_widget(menu_button)
        layout.add_widget(scroll_view)

        self.add_widget(layout)

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

class SearchExtinction(Screen):
    def __init__(self, **kwargs):
        super(SearchExtinction, self).__init__(**kwargs)
        self.build_ui()

    def build_ui(self):
        # Crear el layout principal
        self.layout = BoxLayout(orientation='vertical')

        # Establecer color de fondo
        with self.layout.canvas.before:
            Color(0.75, 0.84, 0.18, 1)  # Verde claro
            self.rect = Rectangle(size=self.layout.size, pos=self.layout.pos)
            self.layout.bind(size=self._update_rect, pos=self._update_rect)

        # Layout para centrar los inputs
        input_layout = AnchorLayout(anchor_x='center', anchor_y='center')
        input_box = BoxLayout(orientation='vertical', size_hint=(None, None), width=300)

        # Crear los TextInputs
        self.input1 = TextInput(hint_text='Height (cm)', size_hint=(None, None), width=300, height=50)
        self.input2 = TextInput(hint_text='Weight (kg)', size_hint=(None, None), width=300, height=50)
        self.input3 = TextInput(hint_text='Diet Type', size_hint=(None, None), width=300, height=50)
        self.input4 = TextInput(hint_text='Was is alone? y/n', size_hint=(None, None), width=300, height=50)

        # Añadir etiquetas y campos de entrada
        input_box.add_widget(Label(text='Height (cm)', size_hint_y=None, height=30))
        input_box.add_widget(self.input1)
        input_box.add_widget(Label(text='Weight (kg)', size_hint_y=None, height=30))
        input_box.add_widget(self.input2)
        input_box.add_widget(Label(text='Diet Code', size_hint_y=None, height=30))
        input_box.add_widget(self.input3)
        input_box.add_widget(Label(text='Social Solitary', size_hint_y=None, height=30))
        input_box.add_widget(self.input4)

        input_layout.add_widget(input_box)
        self.layout.add_widget(input_layout)

        # Etiqueta para mostrar el resultado de la predicción
        self.result_label = Label(text='', size_hint_y=None, height=50)
        self.layout.add_widget(self.result_label)

        # Botón para calcular
        calculate_button = Button(text="Calculate", size_hint=(None, None), width=100, height=50)
        calculate_button.bind(on_release=self.calculate_values)
        self.layout.add_widget(calculate_button)

        # Botón para volver atrás
        back_button = Button(text="Back", size_hint=(None, None), width=100, height=50)
        back_button.bind(on_release=self.go_back)
        self.layout.add_widget(back_button)

        # Añadir el layout a la pantalla
        self.add_widget(self.layout)

    def calculate_values(self, instance):
        h = self.input1.text
        w = self.input2.text
        d = self.input3.text
        s = self.input4.text
        if(s.lower() == 'y') : s = 1 
        elif(s.lower() == 'n') : s = 0
        
        if d in diet_map:
            prediction_result = prediction(h, w, d, s)
            if prediction_result == 0:
                self.result_label.text = "Prediction -> It is not endangered!"
            elif prediction_result == 1:
                self.result_label.text = "Prediction -> It is endangered!"
        else:
            self.result_label.text = "Diet not available!"

    def go_back(self, instance):
        self.manager.current = 'main_screen'  # Cambiar de pantalla

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

class SearchScreen(Screen):
    def __init__(self, **kwargs):
        super(SearchScreen, self).__init__(**kwargs)
        self.build_ui()

    def build_ui(self):
        layout = BoxLayout(orientation='vertical')
      
        with layout.canvas.before:
            Color(0.75, 0.84, 0.18, 1)  # Establecer el color de fondo en verde
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
            layout.bind(size=self._update_rect, pos=self._update_rect)

        # Crear un TextInput para la búsqueda
        self.search_input = TextInput(hint_text='Buscar...', size_hint=(None, None), width=300, height=50)
        layout.add_widget(self.search_input)
        
        # Crear un botón para realizar la búsqueda
        search_button = Button(text='Buscar', size_hint=(None, None), width=100, height=50)
        search_button.bind(on_release=self.store_search)
        layout.add_widget(search_button)
        
        # Crear un área para mostrar los resultados
        self.results_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=500)
        self.results_layout.bind(minimum_height=self.results_layout.setter('height'))

        self.scroll_view = ScrollView()
        self.scroll_view.add_widget(self.results_layout)
        layout.add_widget(self.scroll_view)

        # Crear un botón para volver atrás
        back_button = Button(text="Back", size_hint=(None, None), width=100, height=50)
        back_button.bind(on_release=lambda x: setattr(self.manager, 'current', 'main_screen'))
        layout.add_widget(back_button)

        self.add_widget(layout)

    def store_search(self, instance):
        search = self.search_input.text
        self.results_layout.clear_widgets()

        # Mostrar el texto almacenado
        search_label = Label(text=f'Search: {search}', size_hint_y=None, height=50)
        self.results_layout.add_widget(search_label)
        
        filtered_df = filter1(search, df_full)
        
        # Mostrar cada columna por separado
        for column in filtered_df.columns:
            column_label = Label(text=f'{column}:', size_hint_y=None, height=30, bold=True)
            self.results_layout.add_widget(column_label)
            
            for value in filtered_df[column]:
                value_label = Label(text=str(value), size_hint_y=None, height=30)
                self.results_layout.add_widget(value_label)
        
        print(filtered_df)
        

        
    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

class ColorScreen(Screen):
    def __init__(self, color, **kwargs):
        super(ColorScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        
        with layout.canvas.before:
            Color(*color)  # Establecer el color de fondo
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
            layout.bind(size=self._update_rect, pos=self._update_rect)
        
        # Crear un botón para volver atrás
        back_button = Button(text="Back", size_hint=(None, None), width=100, height=50)
        back_button.bind(on_release=lambda x: setattr(self.manager, 'current', 'main_screen'))
        
        # Añadir el botón al layout
        layout.add_widget(back_button)
        
        self.add_widget(layout)

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main_screen'))
        sm.add_widget(SearchScreen(name='search_screen'))
        sm.add_widget(AddAnimal(name='orange_screen'))
        sm.add_widget(AnimalClassifierScreen(name='red_screen'))  # Rojo
        sm.add_widget(SearchExtinction(name='green_screen'))  # Verde
        
        return sm

if __name__ == '__main__':
    MyApp().run()