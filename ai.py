from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough

def get_summary(text):
    """
    Generate a summary of the given text
    """
    llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """"
The provided text is a transcription of a conversation.
Your task is to summarize the conversation in a concise manner.
Make sure to include the main points and any important details.
Avoid unnecessary details and keep the summary brief.
Use bullet points if necessary.
Don't include any personal opinions or interpretations.
Make sure to include the speaker names and their respective statements.
Don't write any preamble or conclusion.
Write the Output in the same language as the Input text provided.
        """),
        ("human", """
# Input:
{text}

# Output:
"""        )
    ])
    chain = ({"text": RunnablePassthrough()} | prompt | llm)
    return chain.invoke({"text": text}).content

def get_clean(text):
    """
    Cleanup the given text
    """
    llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """"
The provided text is a transcription of a conversation.
Your task is to clean it up fixing indentation, punctuation and misspelled words.
Make sure to include the speaker names and their respective statements.
Don't write any preamble or conclusion.
Write the Output in the same language as the Input text provided.
        """),
        ("human", """
# Input:
{text}

# Output:
"""        )
    ])
    chain = ({"text": RunnablePassthrough()} | prompt | llm)
    return chain.invoke({"text": text}).content
