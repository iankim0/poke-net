import streamlit as st
from PIL import Image

from model.inference import build_model, load_bundle, predict

BUNDLE_PATH = "export/pokemon_species_id_bundle.pt"
TOP_K = 3

st.title("Pokémon Species ID")
st.caption(
    "Upload an image and see the model's top-k species guesses. "
    "The model places the image into similarity space."
    " Some images will be incorrectly identified."
)


@st.cache_resource
def get_model():
    bundle = load_bundle(BUNDLE_PATH)
    encoder, head = build_model(bundle)
    return bundle, encoder, head


bundle, encoder, head = get_model()

uploaded = st.file_uploader("Upload a Pokémon image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=300)

    results = predict(image, encoder, head, bundle, k=TOP_K)

    st.subheader(f"Top-{TOP_K} predictions")
    for rank, r in enumerate(results, start=1):
        gap = f"  (+{r['gap_to_next']:.4f} over next)" if r["gap_to_next"] is not None else ""
        st.write(f"**{rank}. {r['species']}** — {r['score']:.4f}{gap}")
