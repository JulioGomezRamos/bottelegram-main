import telebot
from telebot import types
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import os
import requests

#Conexion con nuestro BOT

TOKEN = '7256278424:AAG4GD3Ja2ENfTLJ0Zj5F7rD4-CMiqLPoCs'

bot = telebot.TeleBot(TOKEN)
BASE_URL = 'http://api.openweathermap.org/data/2.5/weather?'

# Inicializar el modelo de InceptionV3 para clasificación de imágenes
model = InceptionV3(weights='imagenet')

# Directorio para almacenar las fotos de los usuarios
if not os.path.exists('user_photos'):
    os.makedirs('user_photos')

# Base de datos ficticia de productos categorizados
products_db = {
    "trench_coat": [
        {"id": 401, "imagen": "https://th.bing.com/th/id/OIP._t99-vuCpGduSyDOvDMpJwHaMJ?rs=1&pid=ImgDetMain", "precio": 129.99},
        {"id": 402, "imagen": "https://th.bing.com/th/id/OIP.uT-20CJpTc59c_ZbyfmfSwHaKY?rs=1&pid=ImgDetMain", "precio": 139.99},
        {"id": 403, "imagen": "https://th.bing.com/th/id/R.581c55d84719a8f5cba5078310e48236?rik=HJPQ5l1XS7beTQ&pid=ImgRaw&r=0", "precio": 149.99}
    ],
    "jersey": [
        {"id": 501, "imagen": "https://th.bing.com/th/id/OIP.2g6f7d7lDxL6KjEmC8tTHgHaHa?rs=1&pid=ImgDetMain", "precio": 59.99},
        {"id": 502, "imagen": "https://images.tcdn.com.br/img/img_prod/275189/camisa_branca_com_manga_azul_royal_100_poliester_para_sublimacao_p_2729_1_20200722155023.jpg", "precio": 69.99},
        {"id": 503, "imagen": "https://th.bing.com/th/id/OIP.2ipXRQNR836NdcfgVTXtogHaHM?rs=1&pid=ImgDetMain", "precio": 79.99}
    ],
    "Loafer": [
        {"id": 601, "imagen": "https://th.bing.com/th/id/OIP.koIQvMKc6Ag2PLxZYexbHQHaHa?rs=1&pid=ImgDetMain", "precio": 89.99},
        {"id": 602, "imagen": "https://th.bing.com/th/id/OIP.sWcCxNQAtI44h8UYZNrVygAAAA?rs=1&pid=ImgDetMain", "precio": 99.99},
        {"id": 603, "imagen": "https://th.bing.com/th/id/OIP.RkjnErBE55HMPiVQxbjSvAAAAA?w=400&h=500&rs=1&pid=ImgDetMain", "precio": 109.99}
    ],
    "jean": [
        {"id": 701, "imagen": "https://th.bing.com/th/id/R.1d2a705b247b7deaac8d8dafc14d15a8?rik=HhUNYzKWRNEEsQ&pid=ImgRaw&r=0", "precio": 49.99},
        {"id": 702, "imagen": "https://th.bing.com/th/id/OIP.L1cNT0x4-2Q096HTAkJBgAAAAA?rs=1&pid=ImgDetMain", "precio": 59.99},
        {"id": 703, "imagen": "https://http2.mlstatic.com/pantalon-casual-hombre-slim-fit-varios-colores-D_NQ_NP_962434-MLM31225309221_062019-F.jpg", "precio": 69.99}
    ],
    "running_shoe": [
        {"id": 801, "imagen": "https://i.pinimg.com/originals/f4/bd/cc/f4bdcc873bee50a7243851567ee6be5a.jpg", "precio": 79.99},
        {"id": 802, "imagen": "https://th.bing.com/th/id/R.e3de06e4e30d57be9c0591e392b86e33?rik=nxICboIQpPX78w&pid=ImgRaw&r=0", "precio": 89.99},
        {"id": 803, "imagen": "https://th.bing.com/th/id/OIP.zoR6Aj2QRXWEP3ysfbg6SgHaHa?rs=1&pid=ImgDetMain", "precio": 99.99}
    ],
    "cardigan": [
        {"id": 901, "imagen": "https://th.bing.com/th/id/R.02ddea183525f3228f2ab91907c55063?rik=10qqad7yI3iSiw&pid=ImgRaw&r=0", "precio": 149.99},
        {"id": 902, "imagen": "https://th.bing.com/th/id/OIP.uu8ojp54_SD1KTrarjHiJgHaLH?w=1920&h=2880&rs=1&pid=ImgDetMain", "precio": 159.99},
        {"id": 903, "imagen": "https://th.bing.com/th/id/R.9b2f0ada19404e6ce16af500b0415a4f?rik=K68%2b7mgYiJlwng&pid=ImgRaw&r=0", "precio": 169.99}
    ],    
    "cowboy_boot": [
        {"id": 901, "imagen": "https://th.bing.com/th/id/R.3b0c7219b4cd0c17563ff9f380d31e3a?rik=DEzF2Ki8hKO1eg&riu=http%3a%2f%2fimage.sportsmansguide.com%2fadimgs%2fl%2f2%2f281353_ts.jpg&ehk=CIjPUkLJMUJTmgGPQb35Kh9%2bo%2fDkTn%2f%2fZQUU6Vg5bdI%3d&risl=&pid=ImgRaw&r=0", "precio": 149.99},
        {"id": 902, "imagen": "https://th.bing.com/th/id/R.98901ef7e859b1a67ac689e062dca2ce?rik=8gniOcqKkikgrw&pid=ImgRaw&r=0", "precio": 159.99},
        {"id": 903, "imagen": "https://www.oldrebelboots.com/wp-content/uploads/2020/04/Lucchese-1883-Brown-Leather-Cowboy-Western-Boots-Snip-Toe-Vintage-US-Made-Mens-12.jpg", "precio": 169.99}
    ],
    "swimming_trunks": [
        {"id": 1401, "imagen": "https://th.bing.com/th/id/OIP.eYZV1eH0oeJzoicyEXxcuAHaJY?w=990&h=1255&rs=1&pid=ImgDetMain", "precio": 49.99},
        {"id": 1402, "imagen": "https://cdn.laredoute.com/products/8/4/e/84e4f42cb9a03cc2c49365ad4b713970.jpg?width=700&dpr=1", "precio": 59.99},
        {"id": 1403, "imagen": "https://cdn.shopify.com/s/files/1/1924/7769/products/Folpetto_0010_gileslightblue_1200x.jpg?v=1643547187", "precio": 69.99}
    ]
}
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_image(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=1)[0]
    label = decoded_preds[0][1]  # Obtiene la etiqueta de la predicción
    return label

def recommend_products(category):
    # Filtrar productos por categoría
    if category in products_db:
        return products_db[category]
    else:
        return []

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, '¡Hola! Soy tu asistente de compras de ropa de hombre. Envía una foto del producto que deseas y te sugeriré algunos productos similares.')

@bot.message_handler(commands=['help'])
def send_help(message):
    bot.reply_to(message, 'Puedes interactuar conmigo enviando una foto de un producto y te sugeriré productos similares.')

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        user = message.from_user
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        photo_path = f'user_photos/{user.id}.jpg'
        
        with open(photo_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        
        bot.reply_to(message, 'Foto recibida. Procesando...')
        category = classify_image(photo_path)
        bot.reply_to(message, f'Categoría detectada: {category}. Estos productos podrían interesarte:')
        
        recommended_products = recommend_products(category)
        
        if recommended_products:
            for product in recommended_products:
                product_id = product["id"]
                image_url = product["imagen"]
                price = product["precio"]
                
                bot.send_photo(message.chat.id, image_url, caption=f'Producto ID: {product_id}\nPrecio: Q{price}')
        else:
            bot.reply_to(message, 'Lo siento, no encontré productos similares en nuestra base de datos.')
    except Exception as e:
        bot.reply_to(message, f'Hubo un error al procesar la imagen: {str(e)}')

if __name__ == "__main__":
    bot.polling(none_stop=True)