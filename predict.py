import sys

from PIL import Image

from model.inference import build_model, load_bundle, predict

BUNDLE_PATH = "export/pokemon_species_id_bundle.pt"
TOP_K = 3

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    bundle = load_bundle(BUNDLE_PATH)
    encoder, head = build_model(bundle)

    image = Image.open(image_path)
    results = predict(image, encoder, head, bundle, k=TOP_K)

    print(f"\ntop-{TOP_K} predictions for {image_path}:")
    for rank, r in enumerate(results, start=1):
        gap = f"  (+{r['gap_to_next']:.4f} over next)" if r["gap_to_next"] is not None else ""
        print(f"  {rank}. {r['species']:<20} {r['score']:.4f}{gap}")
