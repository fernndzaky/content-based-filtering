import csv
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

def safe_literal_eval(s):
    try:
        return literal_eval(s)
    except SyntaxError:
        return []  # or some other default value

@app.route('/recommend-recipes', methods=['POST'])
def recommend_recipes():
    csv_file_path = 'Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv'  # Replace with the path to your CSV file

    # Retrieve the input ingredients from the request
    input_ingredients = request.json['ingredients']

    # Read the first 5 rows from the CSV file and extract the title and ingredients
    titles = []
    ingredients = []

    with open(csv_file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            titles.append(row['Title'])
            ingredients.append(row['Ingredients'])

    # Convert ingredients from string to list
    ingredients = [safe_literal_eval(ingredient) for ingredient in ingredients]

    # Vectorize the ingredients using TF-IDF
    vectorizer = TfidfVectorizer()
    ingredient_vectors = vectorizer.fit_transform([' '.join(ingredient) for ingredient in ingredients])

    # Vectorize the input ingredients
    input_vector = vectorizer.transform([' '.join(input_ingredients)])

    # Calculate the cosine similarity between input vector and ingredient vectors
    similarities = cosine_similarity(input_vector, ingredient_vectors).flatten()

    # Take only scores above 0
    sorted_similarities = similarities[(similarities > 0)]

    # Sort the recipes based on similarity scores
    sorted_indices = sorted_similarities.argsort()[::-1]

    # Prepare the recommended recipes as a list of dictionaries
    recommended_recipes = []
    for index in sorted_indices[:4]:
        recommended_recipes.append({
            'title': titles[index],
            'ingredients': ingredients[index]
        })

    # Return the recommended recipes as a JSON response
    return jsonify(recommended_recipes)

if __name__ == '__main__':
    app.run()
