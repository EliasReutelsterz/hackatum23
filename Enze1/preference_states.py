class PreferenceStates:
    def __init__(self):
        self.allergies: list[str] = []
        self.utensils: list[str] = []
        self.votes: list[dict] = []
    
    def add_preference_vote(self, vote: dict[str, str]):
        self.votes.append(vote)
    
    def add_allergy(self, allergy: str):
        self.allergies.append(allergy)

    def get_tags(self):
        return ["Family Friendly", "Mediterranean"]
    