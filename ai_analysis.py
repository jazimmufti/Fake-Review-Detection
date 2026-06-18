from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)
prompt = PromptTemplate(
    input_variables=[
        "review",
        "prediction",
        "confidence"
    ],
    template="""
You are an expert review authenticity analyst.

Review:
{review}

Model Prediction:
{prediction}

Model Confidence:
{confidence}%

Analyze the review and explain:

1. Why the model likely predicted this class.
2. Linguistic patterns observed.
3. Indicators suggesting authenticity or deception.
4. Any limitations of the prediction.
5. A brief final conclusion.

Write in clear paragraphs.
Maximum 120 words.
Do not repeat the review text.
"""
)



def generate_ai_analysis(review, prediction, confidence):
    try:
        final_prompt = prompt.format(
            review=review,
            prediction=prediction,
            confidence=confidence
        )

        response = llm.invoke(final_prompt)

        return response.content

    except Exception as e:
        print(f"Gemini Error: {e}")
        return "AI analysis could not be generated at this time."

