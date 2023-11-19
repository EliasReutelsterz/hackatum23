from typing import Union
from pathlib import Path
import secrets

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from recipe_retriever import RecipeRetriever
from preference_states import PreferenceStates
from preference_selection_config import allergies, utensils, preference_mapping

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = str(BASE_DIR / "templates")
STATIC_DIR_FROM_TEMPLATE = "../static"
STATIC_DIR = str(BASE_DIR / "static")

app = FastAPI()

secret_key = secrets.token_urlsafe(32)
app.add_middleware(SessionMiddleware, secret_key=secret_key)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory="../templates")

rr = RecipeRetriever()
ps = PreferenceStates()

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dietary-restrictions", name="dietary-restrictions")
def dietary_restrictions(request: Request):
    return templates.TemplateResponse("dietary_restrictions.html", {"request": request, "allergy_buttons": allergies, "utensils_buttons": utensils})

@app.post("/dietary-restrictions/log", name="process_dietary_restrictions")
def process_dietary_restrictions(result: dict):
    allergies = result.get("allergy", [])
    utensils = result.get("utensils", [])
    preferences = []
    for allergy in allergies:
        if allergy not in preference_mapping:
            continue
        preferences.extend(preference_mapping[allergy])
    preferences = list(set(preferences))
    return {"status": "success"}

@app.get("/personal-preferences")
async def get_personal_preferences(request: Request):
    recipes = rr.draw_recipes(ps.allergies)
    return templates.TemplateResponse("personal_preferences.html", {"request": request, "recipes": recipes})

@app.post("/personal-preferences/log")
def log_personal_preference(result: dict):
    ps.add_preference_vote(result)
    return {"status": "success"}

@app.get("/recipes/draw")
def draw_recipes():
    recipe_list = rr.draw_recipes([])
    return recipe_list

@app.get("/recipes", name="recipes")
def recipes(request: Request):
    tags = ps.get_tags()
    con_input = ps.votes
    allergies = ps.allergies

    recipes = rr.get_customized_recipes(tags, con_input, 15, allergies)
    print(recipes)

    return templates.TemplateResponse("recipes.html", {"request": request, "recipes": recipes})

@app.post("/recipes", name="process-recipes")
async def process_recipes(request: Request, recipes: str = Form(...), people_count: int = Form(...)):
    request.session["recipes"] = recipes
    request.session["people_count"] = people_count
    request.session["message"] = "Order placed successfully!"
    return RedirectResponse(url=request.url_for('success'), status_code=303)

@app.get("/success", name="success")
def finish(request: Request):
    message = request.session.get("message", "")
    return templates.TemplateResponse("success.html", {"request": request, "message": message})

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

