# from smolagents import CodeAgent, HfApiModel

# from polaris.tools import retriever

from uuid import uuid5, NAMESPACE_URL


def main (model_name: str) -> None:
    # model = HfApiModel(model_name)
    # agent = CodeAgent(tools=[], model=model)

    # agent.run()
    path = "src/polaris/tools/retriever.jpg"
    print(uuid5(NAMESPACE_URL, path), NAMESPACE_URL)
    path = "src/polaris/tools/retriever_0.jpg"
    print(uuid5(NAMESPACE_URL, path), NAMESPACE_URL)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
    
    args = parser.parse_args()

    main(args.model)