import csv
import pickle
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
    csv_file_path = 'cleaned_dataset.csv'  # Replace with the path to your CSV file

    # Retrieve the input cleaned_ingredients from the request
    input_ingredients = request.json['ingredients']
    input_course = request.json['course']
    input_cuisine = request.json['cuisine']

    # Read the first 5 rows from the CSV file and extract the title and cleaned_ingredients
    titles = []
    cleaned_ingredients = []
    ingredients = []
    instructions = []
    courses = []
    cuisines = []

    with open(csv_file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            titles.append(row['name'])
            cleaned_ingredients.append(row['cleaned_ingredients'])
            ingredients.append(row['ingredients'])
            instructions.append(row['url'])
            courses.append(row['course'])
            cuisines.append(row['cuisine'])

    # Convert cleaned_ingredients from string to list
    cleaned_ingredients = [safe_literal_eval(ingredient) for ingredient in cleaned_ingredients]

    # Vectorize the cleaned_ingredients using TF-IDF
    vectorizer = TfidfVectorizer()
    ingredient_vectors = vectorizer.fit_transform([' '.join(ingredient) for ingredient in cleaned_ingredients])

    # Vectorize the input cleaned_ingredients
    input_vector = vectorizer.transform([' '.join(input_ingredients)])

    # Calculate the cosine similarity between input vector and ingredient vectors
    similarities = cosine_similarity(input_vector, ingredient_vectors).flatten()

    # Sort the recipes based on similarity scores
    sorted_indices = similarities.argsort()[::-1]

    # Prepare the recommended recipes as a list of dictionaries
    recommended_recipes = []
    for index in sorted_indices[:4]:
        if (input_course.lower() == 'anything' or courses[index].lower() == input_course.lower()) and \
                (input_cuisine.lower() == 'anything' or cuisines[index].lower() == input_cuisine.lower()):
            recommended_recipes.append({
                'title': titles[index],
                'ingredients': ingredients[index],
                'instructions': instructions[index],
                'courses': courses[index],
                'cuisines': cuisines[index],
                'input_ingredients': input_ingredients,
            })

    # Return the recommended recipes as a JSON response
    return jsonify(recommended_recipes)

if __name__ == '__main__':
    app.run()
