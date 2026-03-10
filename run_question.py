import asyncio
import logging
import dotenv

from bot.my_bot import MyBot
from forecasting_tools import MetaculusClient


dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)


QUESTION_URLS = [
    "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
]


async def main() -> None:
    bot = MyBot()
    client = MetaculusClient()

    questions = [
        client.get_question_by_url(url)
        for url in QUESTION_URLS
    ]

    reports = await bot.forecast_questions(questions)

    for report in reports:
        print("\n" + "=" * 100)
        print(f"Question: {report.question.question_text}")
        print(f"URL: {report.question.page_url}")
        print(f"Prediction: {report.prediction}")
        print(f"Cost estimate: {report.price_estimate}")
        print(f"Minutes taken: {report.minutes_taken}")
        print(f"Errors: {report.errors}")

        print("\nExplanation:\n")
        explanation = report.explanation or "(No explanation returned)"
        print(explanation[:4000])

        if len(explanation) > 4000:
            print("\n[Explanation truncated]")


if __name__ == "__main__":
    asyncio.run(main())