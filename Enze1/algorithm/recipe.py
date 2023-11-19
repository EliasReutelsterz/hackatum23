from .ingredient import Ingredient

class Recipe:
    def __init__(self, name: str, ingredients: list[Ingredient], total_time: int, difficulty: int, allergens: list[str]):
        self.name = name
        self.ingredients = ingredients
        self.entropy = 0
        self.total_time = total_time
        self.difficulty = difficulty
        self.allergens = allergens