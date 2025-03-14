from smolagents import Tool

# from polaris.vectorstore import retriever


class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)

    def forward(self, query: str, top_k: int = 4) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        # docs = self.retriever.invoke(
        #     query,
        # )
        # return "\nRetrieved documents:\n" + "".join(
        #     [
        #         f"\n\n===== Document {str(i)} =====\n" + doc.page_content
        #         for i, doc in enumerate(docs)
        #     ]
        # )
