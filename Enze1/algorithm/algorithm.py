import numpy as np
import pandas as pd
#import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from .ingredient import Ingredient
from algorithm.recipe import Recipe
from pathlib import Path
from .categories import Categories
import warnings
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance
import plotly.graph_objects as go
import random



warnings.filterwarnings('ignore')

res_path = Path(__file__).parents[2] / "data" /"recipes-us.json"

class Algorithm:
    
    def get_data(self):
        recipes = {}
        ingredients = {}
        df = pd.read_json(res_path)
        # Find recipes for which the ID is missing or empty
        empty_ids = df[df["ID"] == ""]
        # Drop these rows from the DataFrame
        df.drop(empty_ids.index, inplace=True)
        # Reindex the DataFrame
        df.reset_index(drop=True, inplace=True)
        #print(f"Shape after dropping: {df.shape}")
        #print(df.columns)

        df["Ingredients"].loc[0][0].keys()
        # Count number of unique ingredients
        ingredients_per_recipe = df["Ingredients"].apply(lambda x: [y["Type"] for y in x])
        flat_ingredients = pd.Series([item for sublist in ingredients_per_recipe for item in sublist])
        flat_ingredients.value_counts()


        # Look at ingredient families
        ingredient_families = df["Ingredients"].apply(lambda x: [y["Family"]["Type"] for y in x])
        # Flatten the list of ingredient families
        flat_families = pd.Series([item for sublist in ingredient_families for item in sublist])
        for ingredient in flat_families.value_counts().keys():
            ingredients[ingredient] = Ingredient(ingredient)

        
        #fill recipes
        tags_to_recipe = {}
        for i in range(len(df)):
            recipe = df.iloc[i]
            allergens = []
            for entry in recipe["Allergens"]:
                allergens.append(entry["Name"])
            recipes[recipe["ID"]] = Recipe(name=recipe["Name"], ingredients=[Ingredient(y["Family"]["Type"]) for y in recipe["Ingredients"]], total_time=recipe["TotalTime"], difficulty=recipe["Difficulty"], allergens=allergens)
            for tag in recipe["Tags"]:
                for preference in tag["Preferences"]:
                    if preference in tags_to_recipe.keys():
                        tags_to_recipe[preference].append(recipe["ID"])
                    else:
                        tags_to_recipe[preference] = [recipe["ID"]]
                

        #calculate ingredient occurences
        for recipe in recipes.values():
            for ingredient in recipe.ingredients:
                ingredients[ingredient.name].recipe_occurences += 1

        for ingredient in ingredients.values():
            #!think about taking -log but first of all we are going 0 to 1
            ingredient.entropy = ingredient.recipe_occurences / len(recipes.keys())
        for recipe in recipes.values():
            for ingredient in recipe.ingredients:
                recipe.entropy += ingredients[ingredient.name].entropy
        
        return ingredients, recipes, tags_to_recipe

    


    def create_recipe_ingredient_matrix(self, ingredients, recipes):
        # Step 1: Create a matrix filled with zeros
        matrix = {recipe: {ingredient.name: 0 for ingredient in ingredients.values()} for recipe in recipes.keys()}

        # Step 2: Fill the matrix with ones where appropriate
        for recipe in recipes.keys():
            for ingredient in recipes[recipe].ingredients:
                matrix[recipe][ingredient.name] = 1
        # Step 3: Convert the matrix to a DataFrame
        df = pd.DataFrame(matrix)
        return df
    
    def calculate_general_preference_profile(self, df_ingredient_matrix, tags_to_recipe, tags_selected): #TODO input tags_selected needed from website
        potential_liked_recipes = []
        for tag_selected in tags_selected:
            potential_liked_recipes += tags_to_recipe[tag_selected]
        #!parameter for inmpact of mean preference instead of liked preference
        impact_preference = 4
        df_ingredient_matrix["general_preference_profile"] = df_ingredient_matrix[potential_liked_recipes].sum(axis=1) * impact_preference
        df_ingredient_matrix["general_preference_profile"] /= len(potential_liked_recipes)
        #!parameter for inmpact of mean preference instead of liked preference
        impact = 0.1
        df_ingredient_matrix["general_preference_profile"] += 1 / len(df_ingredient_matrix["general_preference_profile"]) * impact
        df_ingredient_matrix["general_preference_profile"] /= max(df_ingredient_matrix["general_preference_profile"])
        return(df_ingredient_matrix["general_preference_profile"])
    
    def get_row_for_recipe(self, recipes, categories, input_string_recipe, isWinner, ingredients):
        #row means one row in regression output of a dataframe
        input_recipe = recipes[input_string_recipe]
        categories_collection = {}
        for category in categories.keys():
            categories_collection[category] = None
            for ingredient_in_category in categories[category]:
                current_max_entropy = 0
                list_ingredients = {}
                for ingredient in input_recipe.ingredients:
                    list_ingredients[ingredient.name] = ingredients[ingredient.name].entropy
                if ingredient_in_category in list_ingredients.keys():
                    if current_max_entropy < list_ingredients[ingredient_in_category]:
                        categories_collection[category] = ingredient_in_category
                        current_max_entropy = list_ingredients[ingredient_in_category]
        
        categories_collection["difficulty"] = recipes[input_string_recipe].difficulty
        categories_collection["total_time"] = recipes[input_string_recipe].total_time
        categories_collection["isWinner"] = 1 if isWinner else 0
        return list(categories_collection.values())


    def conjoint(self, conjoint_inputs, recipes, categories, general_preference_profile, ingredients):
        cols = list(categories.keys())
        cols += ["difficulty", "total_time", "isWinner"]
        df_conjoint = pd.DataFrame(columns=cols)
        for conjoint_input in conjoint_inputs:
            test_list = []
            
            if conjoint_input["winner"] != "":
                #! Is it really empty string?
                df_conjoint.loc[len(df_conjoint.index)] = self.get_row_for_recipe(recipes, categories, conjoint_input["winner"], True, ingredients)
                

                #df_conjoint.loc[len(df_conjoint.index)] = self.get_row_for_recipe(recipes, categories, conjoint_input["winner"], True, ingredients) #its red but it works
            for loser in conjoint_input["loser"]:
                df_conjoint.loc[len(df_conjoint.index)] = self.get_row_for_recipe(recipes, categories, loser, False, ingredients)
                
                #df_conjoint.loc[len(df_conjoint.index)]= self.get_row_for_recipe(recipes, categories, loser, False, ingredients) #its red but it works

                #TODO fallunterscheidung, glaube außerhalb der schleife, außerdem dann winner und looser getrennt -> concatenate das ganze
                
        winning_ratios = {}
        categories = {}

        #!total_time and difficulty still missing ... not included

        for column in df_conjoint.columns[:11]:
            total = df_conjoint[column].value_counts()
            winners = df_conjoint[df_conjoint['isWinner'] == 1][column].value_counts()
            winning_ratio = winners / total
            winning_ratios.update(winning_ratio.to_dict())
            
            ingredients = df_conjoint[column].unique()
            for ingredient in ingredients:
                categories[ingredient] = column

        df_winning_ratios = pd.DataFrame.from_dict(winning_ratios, orient='index', columns=['winning_ratio'])
        df_winning_ratios['category'] = df_winning_ratios.index.map(categories)
        integer_indices = [index for index in df_winning_ratios.index if isinstance(index, int)]
        df_winning_ratios = df_winning_ratios.drop(integer_indices)

        df_regression = df_conjoint.copy()
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        df_encoded = pd.DataFrame(encoder.fit_transform(df_regression.drop('isWinner', axis=1)))
        df_encoded.columns = encoder.get_feature_names_out(df_regression.drop('isWinner', axis=1).columns)
        y = df_regression['isWinner']
        X = df_encoded
        model = LinearRegression()
        model.fit(X, y)
        #predictions = model.predict(X)

        min_coefficient = min(model.coef_)
        max_coefficient = max(model.coef_)
        normalized_coefficients = (model.coef_ - min_coefficient) / (max_coefficient - min_coefficient)

        category_coefficients = dict(zip(df_regression.drop('isWinner', axis=1).columns, normalized_coefficients))
        df_winning_ratios['multiplied_by_coefficient'] = df_winning_ratios.apply(lambda row: row['winning_ratio'] * category_coefficients[row['category']], axis=1)
        df_winning_ratios = df_winning_ratios.fillna(0)
        min_value = df_winning_ratios['multiplied_by_coefficient'].min()
        max_value = df_winning_ratios['multiplied_by_coefficient'].max()
        #! falls updates zu groß /zu klein ändere diesen parameter ab
        impact = 1
        df_winning_ratios['multiplied_by_coefficient'] = ((df_winning_ratios['multiplied_by_coefficient'] - min_value) / (max_value - min_value) - 0.5) * impact 
        df_winning_ratios.drop('winning_ratio', axis=1, inplace=True)
        for row in df_winning_ratios.index:
            general_preference_profile[row] += df_winning_ratios.loc[row, 'multiplied_by_coefficient']
        general_preference_profile -= min(general_preference_profile)
        general_preference_profile /= max(general_preference_profile)
        return general_preference_profile
        
    def print_pca(self, ingredients, recipes, general_preference_profile, specific_preference_profile):

        # Perform a fit transform on the matrix
        pca = PCA(n_components=3)
        matrix = self.create_recipe_ingredient_matrix(ingredients, recipes)
        matrix["general_preference_profile"] = general_preference_profile
        matrix["specific_preference_profile"] = specific_preference_profile
        matrix["without_any_preference"] = 2 / len(matrix["general_preference_profile"])
        transformed_matrix = pca.fit_transform(matrix)

        # Create a 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=transformed_matrix[:-3, 0],
            y=transformed_matrix[:-3, 1],
            z=transformed_matrix[:-3, 2],
            mode='markers',
            marker=dict(
                size=6,
                color='blue',                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )
        )])

        # Add last three points with different colors
        fig.add_trace(go.Scatter3d(
            x=[transformed_matrix[-3, 0]],
            y=[transformed_matrix[-3, 1]],
            z=[transformed_matrix[-3, 2]],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                opacity=0.8
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=[transformed_matrix[-2, 0]],
            y=[transformed_matrix[-2, 1]],
            z=[transformed_matrix[-2, 2]],
            mode='markers',
            marker=dict(
                size=10,
                color='green',
                opacity=0.8
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=[transformed_matrix[-1, 0]],
            y=[transformed_matrix[-1, 1]],
            z=[transformed_matrix[-1, 2]],
            mode='markers',
            marker=dict(
                size=10,
                color='black',
                opacity=0.8
            )
        ))

        # Set labels
        fig.update_layout(scene = dict(
                            xaxis_title='PC1',
                            yaxis_title='PC2',
                            zaxis_title='PC3'),
                            width=700,
                            margin=dict(r=20, b=10, l=10, t=10))

        fig.show()

    def get_similarities(self, recipes, specific_preference_profile, ingredients):
        matrix = self.create_recipe_ingredient_matrix(ingredients, recipes)
        similarities = pd.DataFrame(0, index=np.arange(1), columns=matrix.columns)
        for recipe_key in recipes.keys():
            similarities[recipe_key] = distance.euclidean(matrix[recipe_key], specific_preference_profile)
        similarities -= min(similarities.iloc[0])
        similarities /= max(similarities.iloc[0])
        return similarities
        
    def filter_allergens(self, recipes, allergens, similarities):
        for allergen in allergens:
            for recipe_key in recipes.keys():
                if allergen in recipes[recipe_key].allergens:
                    if recipe_key in similarities.columns:
                        similarities.pop(recipe_key)

        return similarities

    def get_recommendations(self, recipes, specific_preference_profile, ingredients, input_int, allergens):
        recommendations = []
        similarities = self.get_similarities(recipes, specific_preference_profile, ingredients)
        #sorted_similarities = pd.Series(similarities.iloc[0]).sort_values(ascending=False)
        #print(sorted_similarities)
        similarities = self.filter_allergens(recipes, allergens, similarities)
        recipes_count = len(similarities.iloc[0])
        recommendation_count = 0
        max_recommendations = input_int + 3
        if recipes_count <= max_recommendations:
            return list(similarities.columns)
        while recommendation_count < max_recommendations:
            random_int = random.randint(0, recipes_count - 1)
            prob = similarities.iloc[0][random_int]
            random_choice = random.choices([0, 1], weights=[1-prob, prob], k=1)[0]
            if random_choice == 1:
                recommendations.append(similarities.columns[random_int])
                similarities.pop(similarities.columns[random_int])
                recommendation_count += 1
                recipes_count -= 1
        return recommendations


def calculate_recommendations(tags_selected: list[str], conjoint_inputs: list[dict], recommendation_count: int, allergens: list[str]) -> list[str]:
    algorithm = Algorithm()

    #call once and safe data locally
    ingredients, recipes, tags_to_recipe = algorithm.get_data()

    #call once after selecting tags/preferences
    df_ingredient_matrix = algorithm.create_recipe_ingredient_matrix(ingredients, recipes)
    general_preference_profile = algorithm.calculate_general_preference_profile(df_ingredient_matrix, tags_to_recipe, tags_selected)

    #call once after selecting conjoint choices
    #conjoint choices are a list of dicts with winner and loser -> [{"winner": "recipe_id", "loser": ["recipe_id", "recipe_id", ...]"}...]
    specific_preference_profile = algorithm.conjoint(conjoint_inputs, recipes, Categories.CATEGORIES, general_preference_profile.copy(), ingredients)

    #!not used, just for pca plots
    #algorithm.print_pca(ingredients, recipes, general_preference_profile, specific_preference_profile)


    #call once after selecting input_int and allergens -> probably right after conjoint
    recommendations = algorithm.get_recommendations(recipes, specific_preference_profile, ingredients, recommendation_count, allergens)
    return recommendations