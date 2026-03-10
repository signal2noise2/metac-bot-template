import asyncio
from bot.my_bot import MyBot


TOURNAMENT_ID = "minibench"


async def main() -> None:
    bot = MyBot()
    await bot.forecast_on_tournament(TOURNAMENT_ID)


if __name__ == "__main__":
    asyncio.run(main())