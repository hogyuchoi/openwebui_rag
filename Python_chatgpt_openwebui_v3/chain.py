from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain


def ask_question(docs, question):
    from langchain.chains.llm import LLMChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    # Assume llm and ChatPromptTemplate already defined above...


    llm = ChatOllama(model="mistral", temperature=0.2)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using the following context:\n\n{context}\n\nQuestion: {input}"
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )
    # --- DEBUG ---
    print("DEBUG: docs is type:", type(docs), "first doc:", type(docs[0]) if docs else "None")
    # -------------
    result = chain.invoke({
        "input_documents": docs,    # <- required!
        "input": question           # <- required!
    })
    print("LLM output:", result["output_text"])
    return result["output_text"]

