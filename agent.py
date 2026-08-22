from ip_conceptual_search_agent import generate_synonyms


def main() -> None:
    result = generate_synonyms(
        concept="coordinated multipoint handoff in cellular networks",
        context="H04W CPC Classification area.",
    )
    print(str(result).encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
