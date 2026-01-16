from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger("ai")

def _split_big(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400000,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    chunks = [c.strip() for c in splitter.split_text(text or "")]
    return [c for c in chunks if c]


def _llm():
    model = "gpt-5-mini"
    temperature = 0.2
    logger.debug("Creating ChatOpenAI client (model=%s, temperature=%s)", model, temperature)
    return ChatOpenAI(temperature=temperature, model=model)


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

    started_at = time.monotonic()
    input_len = len(text or "")
    logger.debug("AI pipeline start (input_len=%d)", input_len)

    chunks = _split_big(text)
    if not chunks:
        logger.warning("AI pipeline: empty input")
        return "", "", None

    logger.debug(
        "AI pipeline: split input into %d chunk(s) (chunk_lens=%s)",
        len(chunks),
        [len(c) for c in chunks],
    )

    llm = _llm()

    clean_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are given a multi-speaker transcription where overlapping speech caused sentences to be split into single words.
Speaker labels are alternated line by line, even when the same speaker is continuing the same sentence.

Task:
- Reconstruct readable sentences by merging fragments that logically belong together, even if interrupted by the other speaker.
- Preserve the original wording as much as possible.
- Do NOT paraphrase, summarize, translate, or add content.
- Do NOT change speaker labels.
- Keep the original language.
- Merge fragments when they clearly form a single sentence.
- Preserve punctuation when possible; add minimal punctuation only if required for readability.
- Do NOT add explanations, comments, or preambles.
- Output ONLY the cleaned transcription.
- Fix transcription errors only for misunderstood words that are obvious from context.

### Example 1

Input:
Channel 0: Ciao
Channel 1: Ciao
Channel 0: some
Channel 1: tutto bene?
Channel 0: stai?
Channel 1: Io assonnato
Channel 0: Io
Channel 1: ma felice
Channel 0: sto bene, grazie

Output:
Channel 0: Ciao come stai?
Channel 1: Ciao tutto bene?
Channel 0: Io sto bene, grazie
Channel 1: Io assonnato, ma felice

### Example 2 (English)

Input:
Alice: I
Bob: Yes?
Alice: was
Bob: go on
Alice: saying
Bob: okay
Alice: that the system
Bob: sounds boob
Alice: is ready

Output:
Alice: I was saying that the system is ready
Bob: Yes? go on okay sounds good

### Example 3

Input:
Mario Rossi: Non
Maria Bianchi: Sì?
Mario Rossi: ho
Maria Bianchi: dimmi
Mario Rossi: carpito
Maria Bianchi: pure
Mario Rossi: niente

Output:
Mario Rossi: Non ho capito niente
Maria Bianchi: Sì? dimmi pure
                """.strip(),
            ),
            (
                "human",
                """
# Input Transcription:
{text}

# Output:
"""        )
    ])
    chain = prompt | llm
    return chain.invoke({"text": text}).content

def get_clean(text):
    """
    Cleanup the given text
    """
    llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
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
""".strip(),
            ),
        ]
    )
    summarize_chunk_chain = summarize_chunk_prompt | llm

    chunk_summaries = []
    summarize_chunks = _split_big(cleaned)
    logger.debug("AI pipeline: summarizing %d cleaned chunk(s)", len(summarize_chunks))
    for idx, chunk in enumerate(summarize_chunks):
        try:
            logger.debug("AI pipeline: summarizing chunk %d/%d (len=%d)", idx + 1, len(summarize_chunks), len(chunk))
            chunk_summaries.append(summarize_chunk_chain.invoke({"text": chunk}).content)
        except Exception:
            logger.exception("AI pipeline: failed summarizing chunk %d/%d", idx + 1, len(summarize_chunks))
            raise

    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are given bullet summaries of chunks of a conversation.
Merge them into a single concise summary.
- Keep bullet points.
- Keep speaker names if present.
- Maintain factual accuracy.
- Make sure every important point from the chunks is included.
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
    try:
        summary = reduce_chain.invoke({"text": "\n\n".join([s.strip() for s in chunk_summaries if s and s.strip()])}).content
    except Exception:
        logger.exception("AI pipeline: failed reducing chunk summaries")
        raise
    summary = (summary or "").strip()
    logger.debug("AI pipeline: summary_len=%d", len(summary))

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

    try:
        sentiment_text = sentiment_chain.invoke({"text": cleaned[:20000]}).content
    except Exception:
        logger.exception("AI pipeline: failed sentiment scoring")
        raise
    sentiment = None
    if sentiment_text is not None:
        try:
            sentiment = _clamp_sentiment(int(sentiment_text.strip()))
        except Exception:
            sentiment = None

    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    logger.info(
        "AI pipeline done (cleaned_len=%d summary_len=%d sentiment=%s elapsed_ms=%d)",
        len(cleaned),
        len(summary),
        sentiment,
        elapsed_ms,
    )

    return cleaned, summary, sentiment
