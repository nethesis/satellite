from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter

def _split_big(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=12000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    chunks = [c.strip() for c in splitter.split_text(text or "")]
    return [c for c in chunks if c]


def _llm():
    return ChatOpenAI(temperature=0.3, model="gpt-5-mini")


def _clamp_sentiment(value: int) -> int:
    if value < 0:
        return 0
    if value > 10:
        return 10
    return value


def generate_clean_summary_sentiment(text: str):
    """Generate cleaned transcription, summary, and sentiment (0-10).

    Splits very long transcripts into large chunks to stay within model context.
    """

    chunks = _split_big(text)
    if not chunks:
        return "", "", None

    llm = _llm()

    clean_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are given a multi-speaker transcription where overlapping speech caused sentences to be split into single words across consecutive lines of the same speaker.

Task:
- Reconstruct readable sentences by merging consecutive fragments spoken by the same channel.
- Preserve the original wording as much as possible.
- Do NOT paraphrase, summarize, translate, or add content.
- Do NOT change speaker labels.
- Keep the original language.
- If a fragment clearly completes the previous utterance of the same channel, merge it.
- If speakers alternate, do NOT merge across channels.
- Preserve punctuation when possible; add minimal punctuation only if required for readability.
- Do NOT add explanations, comments, or preambles.
- Output ONLY the cleaned transcription.
                """.strip(),
            ),
            (
                "human",
                """
# Input:
{text}

# Output:
""".strip(),
            ),
        ]
    )
    clean_chain = clean_prompt | llm

    cleaned_chunks = []
    for chunk in chunks:
        cleaned_chunks.append(clean_chain.invoke({"text": chunk}).content)
    cleaned = "\n\n".join([c.strip() for c in cleaned_chunks if c and c.strip()]).strip()

    summarize_chunk_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
The provided text is a transcription of a conversation.
Summarize this chunk concisely.
- Use bullet points.
- Do NOT change speaker labels.
- Capture main points and important details.
- No opinions.
- Keep speaker names or labels if present.
- Same language as input.
- Do NOT add explanations, comments, or preambles.
- Output ONLY the summary.
                """.strip(),
            ),
            (
                "human",
                """
# Input chunk:
{text}

# Output:
""".strip(),
            ),
        ]
    )
    summarize_chunk_chain = summarize_chunk_prompt | llm

    chunk_summaries = []
    for chunk in _split_big(cleaned):
        chunk_summaries.append(summarize_chunk_chain.invoke({"text": chunk}).content)

    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are given bullet summaries of chunks of a conversation.
Merge them into a single concise summary.
- Keep bullet points.
- Keep speaker names if present.
- Same language as the summaries.
No preamble or conclusion.
                """.strip(),
            ),
            (
                "human",
                """
{text}
""".strip(),
            ),
        ]
    )
    reduce_chain = reduce_prompt | llm
    summary = reduce_chain.invoke({"text": "\n\n".join([s.strip() for s in chunk_summaries if s and s.strip()])}).content
    summary = (summary or "").strip()

    sentiment_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Rate the overall sentiment expressed in the conversation on a 0-10 scale.
0 means pure hate.
10 means deepest love.

Return ONLY a single integer from 0 to 10.

example output:0
example output:10
example output:5
example output:7
example output:3
                """.strip(),
            ),
            (
                "human",
                """
{text}
""".strip(),
            ),
        ]
    )
    sentiment_chain = sentiment_prompt | llm

    sentiment_text = sentiment_chain.invoke({"text": cleaned[:20000]}).content
    sentiment = None
    if sentiment_text is not None:
        try:
            sentiment = _clamp_sentiment(int(sentiment_text.strip()))
        except Exception:
            sentiment = None

    return cleaned, summary, sentiment