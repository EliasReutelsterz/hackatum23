from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from preference_states import PreferenceStates

import algorithm.algorithm as algorithm

RECIPE_PATH = Path(__file__).parents[1] / "data" / "recipes-us-images.json"

class RecipeRetriever:
    def __init__(self):
        self.INTERESTING_COLUMNS = ["ID", "Name", "Category", "Headline", "Difficulty", "PrepTime", "TotalTime", "ImageLink", "ImagePath", "Slug", "Allergens"]
        self.ALLERGENS = ['Milk', 'Soy', 'Wheat', 'Eggs', 'Tree Nuts', 'Shellfish', 'Fish', 'Sesame', 'Peanuts']
        self.df = None
        self.batches = None

    def prepare_df(self, df, allergies: list[str] = []):
        # Remove empty recipes
        empty_ids = df[df["ID"] == ""]
        # Drop these rows from the DataFrame
        df_dropped = df.drop(empty_ids.index, inplace=False)
        # Reindex the DataFrame
        df_indexed = df_dropped.reset_index(drop=True, inplace=False)

        df_filtered = df_indexed.copy()
        # Remove recipes with matching allergens
        # Get all allergens for each recipe

        allergens_per_recipe = df["Allergens"].apply(lambda x: [y["Type"] for y in x])
        for rec_id, recipe_allergens in allergens_per_recipe.items():
            # Check if any of the recipe allergens are in the user's allergies
            if any(allergen in recipe_allergens for allergen in allergies):
                # Drop the recipe
                df_filtered.drop(rec_id, inplace=True)
        # Reindex the DataFrame
        df_filtered.reset_index(drop=True, inplace=True)
        return df_filtered

    def select_random_recipes(self, n=30) -> list[list[dict]]:
        samples = self.df.sample(n)[self.INTERESTING_COLUMNS].to_dict(orient="records")        

        # Drop duplicates from df
        self.df.drop_duplicates(subset="ID", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        if len(samples) != n:
            print("Warning: Not enough recipes to sample from")
            # Sample remaining recipes and append them to samples
            remaining_samples = self.df.sample(n-len(samples))[self.INTERESTING_COLUMNS].to_dict(orient="records")
            samples.extend(remaining_samples)

        # Batch the samples into groups of 3
        batches = [samples[i:i+3] for i in range(0, len(samples), 3)]
        return batches

    def draw_recipes(self, allergies: list[str]):
        # Check if class is already initialized
        if self.df is None:
            print("Initializing dataframe")
            self.df = self.prepare_df(pd.read_json(RECIPE_PATH), allergies)
        # Check if batches are empty
        if not self.batches:
            print("Loading batches")
            self.batches = self.select_random_recipes()

        # Pop a batch
        if len(self.batches) == 0:
            self.batches = self.select_random_recipes()
        batch = self.batches.pop()

        card_data = [x for x in batch]

        # Fix total time
        for card in card_data:
            if card["TotalTime"] == "":
                if card["PrepTime"] != "":
                    # Remove the "PT" prefix and "M" suffix
                    prepTime = card["PrepTime"][2:-1]
                    card["TotalTime"] = prepTime
                else:
                    card["TotalTime"] = "N/A"
            else:
                # Remove the "PT" prefix and "M" suffix
                totalTime = card["TotalTime"][2:-1]
                card["TotalTime"] = totalTime

        return card_data 
    
    def supplement_image_urls(self):
        # Iterate through the DataFrame and update the ImageLink column
        for index, row in self.df.iterrows():
            # Get the image url
            image_url = self.get_image_url(row)
            # Update the ImageLink column
            self.df.at[index, "ImageLink"] = image_url
        
        # Save the DataFrame to a JSON file
        self.df.to_json(RECIPE_PATH.with_name("recipes-us-images.json"), orient="records")

    def get_image_url(self, recipe) -> str:
        backup_image = "https://img.hellofresh.com/c_fit,f_auto,fl_lossy,h_1100,q_30,w_2600/hellofresh_s3/image/HF220721_R20_W37_DE_VP4181-1_MB_Main_low-ed650078.jpg"

        print(f"Getting image url for {recipe['Name']}")
        # Build url from recipe
        base_url = "https://www.hellofresh.com/recipes/"
        # Get slug
        slug = recipe["Slug"]
        # Get ID
        recipe_id = recipe["ID"]
        # Build url
        url = base_url + slug + "-" + recipe_id
        print(url)

        # Send a GET request to the URL
        try:
            response = requests.get(url, timeout=2)
        except requests.exceptions.Timeout:
            print("Timeout")
            return backup_image

        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the div with the specific data attribute
        div = soup.find('div', attrs={'data-test-id': 'recipe-hero-image'})
        if not div:
            return backup_image
        
        img = div.find('img')
        if img:
            image_url = img.get('src')  # Get the src attribute
            print(image_url)
        else:
            return backup_image
        
        return image_url

    def get_customized_recipes(self, tags_selected, conjoint_inputs, rec_count, allergens):
        id_list = algorithm.calculate_recommendations(tags_selected, conjoint_inputs, rec_count, allergens)
        recipes = self.df[self.df["ID"].isin(id_list)].to_dict(orient="records")
        # Remove duplicates from df
        self.df.drop_duplicates(subset="ID", inplace=True)

        # Update the TotalTime column
        for card in recipes:
            if card["TotalTime"] == "":
                if card["PrepTime"] != "":
                    # Remove the "PT" prefix and "M" suffix
                    prepTime = card["PrepTime"][2:-1]
                    card["TotalTime"] = prepTime
                else:
                    card["TotalTime"] = "N/A"
            else:
                # Remove the "PT" prefix and "M" suffix
                totalTime = card["TotalTime"][2:-1]
                card["TotalTime"] = totalTime
        return recipes