"""
Clustering Application CLI

This script serves as the entry point for running various clustering algorithms
on a specified dataset. 

Supported models include K-Means, DBSCAN, HDBSCAN,
Agglomerative Clustering, Gaussian Mixture Models (GMM), and OPTICS.

Usage:
    python main.py clustering --model kmeans --path "data/data-name-path"
"""
import typer
from pathlib import Path
from kmeans import Kmeans
from dbscan import Dbscan
from hdbscan import Hdbscan
from agglomerative import Agglomerative
from gmm import Gmm
from optics import Optics

app = typer.Typer(help="Clustering Application CLI")

def get_model(model_type: str):
    """
    Retrieve the clustering model instance based on the model type.
    """
    model_mapping = {
        "kmeans": Kmeans,
        "dbscan": Dbscan,
        "hdbscan": Hdbscan,
        "agglomerative": Agglomerative,
        "gmm": Gmm,
        "optics": Optics,
    }

    model_class = model_mapping.get(model_type.lower())
    if model_class is None:
        supported_models = ", ".join(model_mapping.keys())
        raise ValueError(
            f"Unsupported clustering model: '{model_type}'. Supported models are: {supported_models}."
        )
    return model_class()

@app.command()
def main(
    model: str = typer.Option("kmeans", help="Clustering model to use"),
    path: str = typer.Option("data/synthetic/pdw-simulator/pdw.csv", "--path", "-p", help="Path to input data file")
):
    """
    Run clustering analysis with the specified model on the provided dataset.
    """
    try:
        data_path = Path(path)

        print(f"Selected Model: {model}")
        print(f"Data Path: {data_path}")

        if not data_path.is_file():
            print(f"Error: Data file not found at '{data_path}'.")
            raise typer.Exit(code=1)

        model_instance = get_model(model_type=model)

        results = model_instance.run(path=str(data_path))

        print("\nClustering Results:")
        print("-" * 50)
        print(results)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
