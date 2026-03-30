"""Demo: pede uma frase e mostra os resultados"""

import argparse
from attention_viz import run as viz_run


def main():
    parser = argparse.ArgumentParser(description="VetNER demo — enter a sentence to analyze")
    parser.add_argument("--sentence", "-s", type=str, help="Sentença para análise")
    args = parser.parse_args()

    if args.sentence:
        viz_run(args.sentence)
        return

    try:
        sent = input("Digite a sentença para análise: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nEncerrando.")
        return

    if sent:
        viz_run(sent)


if __name__ == "__main__":
    main()
